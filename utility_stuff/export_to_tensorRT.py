from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("front_yolov8n_070424_1.pt")

# Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'
