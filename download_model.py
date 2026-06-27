from ultralytics import YOLO

print("Downloading YOLOv8n model (this may take a moment)...")
model = YOLO("yolov8n.pt")
print("Model downloaded successfully!")
print("Model ready for inference.")
