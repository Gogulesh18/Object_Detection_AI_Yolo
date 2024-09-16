from ultralytics import YOLO
import cv2
model = YOLO('../YOLO-Weights/yolov8l.pt')
result = model("images/img2.png", show=True)
cv2.waitKey(0)