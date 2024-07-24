import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.interpolate import griddata
import os
import random
from ultralytics import YOLO
import supervision as sv

from roboflow import Roboflow
rf = Roboflow(api_key="TNOAfbxZJ9Ol5UAmvyRD")
project = rf.workspace("mars-task").project("box-orientation-detector-lua8h")
version = project.version(3)
dataset = version.download("yolov8-obb")

# import yaml

# with open(f'{dataset.location}/data.yaml', 'r') as file:
#     data = yaml.safe_load(file)

# data['path'] = dataset.location

# with open(f'{dataset.location}/data.yaml', 'w') as file:
#     yaml.dump(data, file, sort_keys=False)

import yaml

# Assuming dataset.location is defined and points to the correct dataset directory
dataset_location = "/content/Box-Orientation-Detector-3"  # Update this as needed

# Read the data.yaml file
yaml_file_path = f'{dataset_location}/data.yaml'
with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file)

# Update the 'path' key in the YAML data
data['path'] = dataset_location

# Write the updated data back to the data.yaml file
with open(yaml_file_path, 'w') as file:
    yaml.dump(data, file, sort_keys=False)

from ultralytics import YOLO

model = YOLO("yolov8n-obb.pt")

results = model.train(data=f"{dataset_location}/data.yaml", epochs=100, imgsz=640)



model = YOLO('runs/obb/train/weights/best.pt')



# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MiDaS model
model_type = "DPT_Large"  # You can also use "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move the model to the appropriate device
midas.to(device)
midas.eval()

# Load the transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Load your YOLO model for box detection
yolo_model = YOLO('runs/obb/train/weights/best.pt')

# Load a random image from your dataset
random_file = random.choice(os.listdir(f"{dataset_location}/test/images"))
file_name = os.path.join(f"{dataset_location}/test/images", random_file)
img = cv2.imread(file_name)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform object detection using YOLO
yolo_results = yolo_model(file_name)
detections = sv.Detections.from_ultralytics(yolo_results[0])

# Transform the image for MiDaS model
input_batch = transform(img_rgb).to(device)

# Perform inference with MiDaS
with torch.no_grad():
    prediction = midas(input_batch)

# Resize and normalize the depth map
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img_rgb.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()

# Raw depth data
raw_depth_map = prediction.cpu().numpy()

# Function to handle missing values by interpolation
def interpolate_missing_values(depth_data):
    h, w = depth_data.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Identify valid (non-missing) values
    valid_mask = depth_data > 0
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]
    valid_z = depth_data[valid_mask]

    # Interpolate missing values
    interpolated_depth = griddata(
        (valid_x, valid_y),
        valid_z,
        (x, y),
        method='linear',
        fill_value=0
    )
    return interpolated_depth

# Interpolate missing values in the depth map
interpolated_depth_map = interpolate_missing_values(raw_depth_map)

# Normalize depth map for visualization
depth_map = cv2.normalize(interpolated_depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map = np.uint8(depth_map)

# Display the depth map
plt.figure(figsize=(10, 10))
plt.imshow(depth_map, cmap='inferno')
plt.title("Depth Map")
plt.axis('off')
plt.show()

# Function to calculate orientation angles from depth data
def calculate_orientation_angle(depth_data):
    h, w = depth_data.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    z = depth_data.flatten()

    # Prepare the data matrix
    A = np.c_[x, y, np.ones(x.shape)]

    # Solve for the plane coefficients (a, b, d)
    C, _, _, _ = lstsq(A, z)

    # The coefficients are the components of the normal vector
    a, b, d = C

    # The normal vector of the plane is (a, b, -1)
    normal_vector = np.array([a, b, -1])

    # Normalize the normal vector
    normal_vector /= np.linalg.norm(normal_vector)

    # Calculate the angles with the axes
    angles = {
        'x': np.arccos(np.abs(normal_vector[0])) * (180.0 / np.pi),
        'y': np.arccos(np.abs(normal_vector[1])) * (180.0 / np.pi),
        'z': np.arccos(np.abs(normal_vector[2])) * (180.0 / np.pi)
    }

    return angles

# Annotate the image with detected boxes
oriented_box_annotator = sv.OrientedBoxAnnotator()
annotated_frame = oriented_box_annotator.annotate(
    scene=img,
    detections=detections
)

# Display the annotated image
sv.plot_image(image=annotated_frame, size=(12, 12))

# Calculate orientation angles for each detected box
for i, box in enumerate(detections.xyxy):
    x1, y1, x2, y2 = map(int, box)

    # Extract depth data for the current box
    box_depth_data = interpolated_depth_map[y1:y2, x1:x2]

    # Calculate orientation angles for the current box
    orientation_angles = calculate_orientation_angle(box_depth_data)
    print(f"Orientation Angles for Box {i+1}: {orientation_angles}")

    # Visualize the depth map for the current box
    box_depth_map = cv2.normalize(box_depth_data, None, 0, 255, cv2.NORM_MINMAX)
    box_depth_map = np.uint8(box_depth_map)

    plt.figure(figsize=(5, 5))
    plt.imshow(box_depth_map, cmap='inferno')
    plt.title(f"Depth Map for Box {i+1}")
    plt.axis('off')
    plt.show()
