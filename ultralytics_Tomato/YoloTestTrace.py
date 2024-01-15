from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt',task='detect')
# model = YOLO('yolov8n-seg.pt')

# Track with the model
results = model.track(source="1.mp4", show=True)
