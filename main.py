from ultralytics import YOLO
model = YOLO('yolov8n.pt')
result = model("images/img1.jpeg", show=True)