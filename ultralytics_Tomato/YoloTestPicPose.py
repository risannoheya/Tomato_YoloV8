from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
results = model('./ultralytics/assets/bus.jpg')
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)
