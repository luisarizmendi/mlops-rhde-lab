from ultralytics import YOLO

model = YOLO("best.pt")  # Replace with your model's path
model.export(format="onnx")

