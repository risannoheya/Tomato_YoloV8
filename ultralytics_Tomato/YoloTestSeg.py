from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('yolov8n-seg.pt')
# Predict with the model
results = model('inside.png')  # predict on an image
res = results[0].plot()
f = open("data.txt")
roi = f.read()
roi=roi.split(',')
a=int(roi[0])
b=int(roi[1])
c=int(roi[2])
d=int(roi[3])
Roi = res[a:b,c:d]
height, width =Roi.shape[:2]
resized_img = cv2.resize(Roi, (width//2, height//2))
cv2.imshow('YOLOv8 Inference', resized_img)
# cv2.imshow('MaxArea',Roi)
f.close()
cv2.waitKey(0)
