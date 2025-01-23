from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Replace with your model's path
model.export(format="onnx")

