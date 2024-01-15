from ultralytics import YOLO
import cv2
# 加载预训练模型
model = YOLO("runs/detect/train/weights/best.pt", task='detect')
results = model("tomato.jpg")
res = results[0].plot()
cv2.imwrite('reslut.jpg',res)

