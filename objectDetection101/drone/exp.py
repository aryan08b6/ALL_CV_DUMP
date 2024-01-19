import cv2
import numpy as np
import socket
import struct
import pickle
import keyboard
import multiprocessing
import time

def receive_frames(client_socket):
    # Set up window for displaying camera feed
    cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

    while True:
        # Receive frame from the Raspberry Pi
        data = b''
        try:
            while True:
                packet = client_socket.recv(4)
                if len(packet) == 0:
                    break
                size = struct.unpack('!I', packet)[0]
                frame_data = client_socket.recv(size)
                data += packet + frame_data
        except Exception as e:
            print("Error receiving frame:", str(e))
            break
        if len(data) == 0:
            break

        frame = pickle.loads(data[4:])

        # Display frame
        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(1)

def send_commands(client_socket):
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
    port = 8888
    client_socket.connect((host, port))

    # Create processes for receiving frames and sending commands
    receive_process = multiprocessing.Process(target=receive_frames, args=(client_socket,))
    send_process = multiprocessing.Process(target=send_commands, args=(client_socket,))

    # Start the processes
    receive_process.start()
    send_process.start()

    # Wait for both processes to finish
    receive_process.join()
    send_process.join()

    cv2.destroyAllWindows()
    client_socket.close()
