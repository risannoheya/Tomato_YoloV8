from ultralytics import YOLO
import cv2
# 加载预训练模型
model = YOLO("runs/detect/train/weights/best.pt", task='detect')
# model = YOLO("yolov8n.pt") task参数也可以不填写，它会根据模型去识别相应任务类别
# 检测图片
results = model("tomato.jpg")
res = results[0].plot()
# f = open("data.txt")
# roi = f.read()
# roi=roi.split(',')
# a=int(roi[0])
# b=int(roi[1])
# c=int(roi[2])
# d=int(roi[3])

# Roi = res[a:b,c:d]
cv2.imwrite('reslut.jpg',res)
# height, width =res.shape[:2]
# resized_img = cv2.resize(res, (width//2, height//2))
# cv2.imshow('YOLOv8 Inference', resized_img)
# # cv2.imshow('MaxArea',Roi)
# # f.close()
# cv2.waitKey(0)
