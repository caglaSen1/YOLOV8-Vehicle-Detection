import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import cvzone

model = YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('video.mp4')

# obj class
coco_file = open("coco.txt", "r")
data = coco_file.read()
class_list = data.split("\n") 
#print(class_list)

count = 0
tracker = Tracker()
cy1 = 300
center_of_line = 510
offset = 6
counter_left = []
counter_right = []


while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    # print(results)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    # print(px)
    obj_list = []

    for index, row in px.iterrows():
        # print(row)
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            obj_list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(obj_list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
            cv2.circle(frame, (cx, cy), 3, (255, 0, 255), -1)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            if cx < center_of_line:
                if counter_left.count(id) == 0:
                    counter_left.append(id)

            elif center_of_line < cx:
                if counter_right.count(id) == 0:
                    counter_right.append(id)

    cv2.line(frame, (0, cy1), (1020, cy1), (0, 255, 0), 2)
    left = (len(counter_left))
    right = (len(counter_right))
    cvzone.putTextRect(frame, f"Counter Left: {left}", (50, 60), 2, 2)
    cvzone.putTextRect(frame, f"Counter Right: {right}", (700, 60), 2, 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

