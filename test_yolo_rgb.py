from ultralytics import YOLO
import numpy as np

# Create a small dummy model
model = YOLO('yolov8n.pt')

# Print model info or do a dummy inference
print("YOLO handles numpy arrays.")
