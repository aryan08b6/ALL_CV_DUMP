from ultralytics import YOLO
import cv2
import math

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

model = YOLO("../Yolo-Weights/yolov8n.pt")
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

print(YOLO.device)

while True:
    success, img = cap.read()
    res = model(img, stream=True)

    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = classNames[int(box.cls[0])]
            conf = (math.ceil(box.conf[0]*100))/100
            if conf > 0.85:
                cv2.putText(img, f'Conf: {conf} \n Class: {cls}', (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)



    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        print(model.device)
        break

cap.release()
