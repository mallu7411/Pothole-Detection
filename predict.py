import os
from ultralytics import YOLO
import cv2
import numpy as np


VIDEO_PATH = os.path.join('.', 'demo pothole 3.mp4')  
OUTPUT_VIDEO_PATH = os.path.join('.', 'outputs')  


model_path = os.path.join('runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)


threshold = 0.5  
class_name_dict = {0: 'pothole'}  

cap = cv2.VideoCapture(VIDEO_PATH)


if not cap.isOpened():
    print(f"Error: Could not open video: {VIDEO_PATH}")
    exit()


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),  
    fps,
    (frame_width, frame_height)
)


while cap.isOpened():
    ret, frame = cap.read()  
    if not ret:
        print("End of video or error reading frame.")
        break

   
    results = model(frame)[0]


    for result in results:
        if result.boxes:
            for box in result.boxes:
                conf = box.conf.item()
                cls_id = box.cls.item()  

                if conf >= threshold:
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                
                    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

                
                    label = f"{class_name_dict.get(cls_id, 'Unknown')} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"Class: {cls_id}, Confidence: {conf}, Box: ({x1}, {y1}, {x2}, {y2})")


    out.write(frame)


    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {OUTPUT_VIDEO_PATH}")