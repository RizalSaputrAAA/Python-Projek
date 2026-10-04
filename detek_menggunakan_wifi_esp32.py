import cv2 as cv
import numpy as np
import socket
import time

# ================= WiFi ESP32 =================
ESP32_IP = "192.168.1.6"
PORT = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((ESP32_IP, PORT))
print("Connected to ESP32")

# ================= Kamera =================
cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Range warna biru
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cx_to_send = 320  # default tengah

    if contours:
        cnt = max(contours, key=cv.contourArea)
        area = cv.contourArea(cnt)

        if area > 500:
            x, y, w, h = cv.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            cx_to_send = cx

            cv.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)
            cv.circle(frame, (cx,cy), 5, (0,0,255), -1)

            frame_h, frame_w, _ = frame.shape
            cv.line(frame, (frame_w//2,0), (frame_w//2,frame_h), (0,255,255), 1)
            cv.line(frame, (frame_w//2,cy), (cx,cy), (255,0,0), 2)

            offset = cx - frame_w//2
            cv.putText(frame, f"Offset: {offset}", (10,30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ================= KIRIM KE ESP32 =================
    try:
        sock.sendall(f"{cx_to_send}\n".encode())
    except:
        print("Connection lost")
        break

    cv.imshow("Camera", frame)
    cv.imshow("Mask", mask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
sock.close()
