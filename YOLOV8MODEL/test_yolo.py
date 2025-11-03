from ultralytics import YOLO
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU device:", torch.cuda.get_device_name(0))

# Load a sample YOLOv8 model
model = YOLO("yolov8n.pt")  # small pre-trained model
print(model)
