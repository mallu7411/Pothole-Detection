import ultralytics 
from ultralytics import YOLO

model = YOLO("yolo11n (1).pt")
results = model.train(data="data.yaml")