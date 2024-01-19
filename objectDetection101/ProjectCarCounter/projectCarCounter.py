import math

import numpy as np
from ultralytics import YOLO
import cv2
from sort import *

model = YOLO("../Yolo-Weights/yolov8l.pt")
cap = cv2.VideoCapture("../Videos/cars.mp4")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")

limits = [400, 297, 673, 297]
totalCount = 0

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    result = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for res in result:
        for bbox in res.boxes:
            x1, y1, x2, y2 = bbox.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(bbox.cls[0])
            conf = (math.ceil(bbox.conf[0]*100))/100
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and bbox.conf[0] > 0.5:
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.putText(img, f'Class: {classNames[cls]}, Conf: {bbox.conf}', (max(0, x1), max(0, y1)), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 0, 0), 2)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 355), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'ID: {Id}', (max(0, x1), max(0, y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cx, cy = x1 + (x2-x1)//2, y1 + (y2-y1)//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        print(result)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            totalCount+=1
            cv2.putText(img, f'count:{totalCount}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0))





    cv2.imshow("Window", img)
    # cv2.imshow("Region", imgRegion)
    if cv2.waitKey(1) == 27:
        print(YOLO.device)
        break

