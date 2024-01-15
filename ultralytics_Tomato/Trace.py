import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')

# 打开视频文件
video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)

# 循环遍历视频帧
while cap.isOpened():
    # 从视频读取一帧
    success, frame = cap.read()

    if success:
        # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
        results = model.track(frame, persist=True)

        # 在帧上展示结果
        annotated_frame = results[0].plot()

        # 展示带注释的帧
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # 如果按下'q'则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则退出循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()