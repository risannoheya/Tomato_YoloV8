from ultralytics import YOLO

def run():
    model = YOLO('yolov8.yaml').load('yolov8n.pt')
    model.train(data='data.yaml', epochs=200, imgsz=640,batchsize=4, save_period=5,device=0)

if __name__=="__main__":
   run()