import cv2
import socket
import struct
import pickle
import keyboard
import multiprocessing
from ultralytics import YOLO
import threading

model = YOLO("../Yolo-Weights/yolov8n.pt")

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


def receive_frames(client_socket):
    # Set up window for displaying camera feed
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

    while True:
        # Receive frame from the Raspberry Pi
        size_data = b''
        while len(size_data) < 4:
            size_data += client_socket.recv(4 - len(size_data))
        size = struct.unpack('!I', size_data)[0]
        data = b''
        while size > 0:
            chunk = client_socket.recv(size)
            if not chunk:
                break
            data += chunk
            size -= len(chunk)
        frame = pickle.loads(data)
        img = frame
        result = model(img, stream=True)
        for res in result:
            for bbox in res.boxes:
                x1, y1, x2, y2 = bbox.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if bbox.conf[0] > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(img, f'Class: {classNames[int(bbox.cls[0])]}, Conf: {bbox.conf}', (max(0, x1), max(0, y1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display frame
        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(1)

def send_commands(client_socket):
    servoAngle = 0
    while True:
        # Check for keyboard input
        if keyboard.is_pressed('up'):
            command = "Forward"
        elif keyboard.is_pressed('down'):
            command = "Backward"
        elif keyboard.is_pressed('left'):
            command = "Left"
        elif keyboard.is_pressed('right'):
            command = "Right"
        elif keyboard.is_pressed('l'):
            command = "Land"
        elif keyboard.is_pressed('t'):
            command = "Takeoff"
        elif keyboard.is_pressed('escape'):
            command = "END"
            client_socket.sendall(command.encode())
            break
        else:
            command = "Stop"

        # Send command to the Raspberry Pi
        client_socket.sendall(command.encode())


if __name__ == '__main__':
    # Set up socket connection
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '192.168.1.100'  # Replace with the Raspberry Pi's IP address
    port = 1234
    client_socket.connect((host, port))

    # Create processes for receiving frames and sending commands
    receive_process = multiprocessing.Process(target=receive_frames, args=(client_socket,))
    send_process = multiprocessing.Process (target=send_commands, args=(client_socket,))

    # Start the processes
    receive_process.start()
    send_process.start()

    # Wait for both processes to finish
    receive_process.join()
    send_process.join()

    cv2.destroyAllWindows()
    client_socket.close()