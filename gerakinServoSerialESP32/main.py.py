import cv2 as cv
import numpy as np
import serial
import time

# Inisialisasi serial
ser = serial.Serial('COM5', 115200)  
time.sleep(2)

cam = cv.VideoCapture(0)

# Dapatkan resolusi frame
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    frame = cv.flip(frame, 1)

    # Konvert Warna Ke HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Range Warna Biru (sesuaikan jika perlu)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Mask
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    
    # Operasi morfologi untuk membersihkan noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    # Hasil
    res = cv.bitwise_and(frame, frame, mask=mask)
    
    # ========== Menampilkan Kotak =================
    # Kontur
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Kirim nilai default ke Arduino jika tidak ada objek terdeteksi
    cx_to_send = 320  # Nilai tengah/default
    
    if len(contours) > 0:
        # Cari kontur dengan area terbesar
        cnt = max(contours, key=cv.contourArea)
        area = cv.contourArea(cnt)

        if area > 500:  # Filter area minimum
            x, y, w, h = cv.boundingRect(cnt)
            
            # Titik tengah objek
            cx = x + w // 2
            cy = y + h // 2
            cx_to_send = cx  # Simpan nilai untuk dikirim ke Arduino
            
            # Kotak Persegi Panjang 
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
            # Gambar garis tengah frame
            frame_h, frame_w, _ = frame.shape
            cv.line(
                frame,
                (frame_w // 2, 0),
                (frame_w // 2, frame_h),
                (0, 255, 255),
                1
            )
            
            # Gambar garis dari tengah frame ke objek
            cv.line(
                frame,
                (frame_w // 2, cy),
                (cx, cy),
                (255, 0, 0),
                2
            )
            
            # Tambahkan teks informasi
            cv.putText(
                frame, 
                "Blue Object",
                (x, y-10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv.putText(
                frame,
                f"Area: {int(area)}",
                (x, y + h + 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
            
            cv.putText(
                frame,
                f"Center: ({cx}, {cy})",
                (x, y + h + 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )
            
            # Tampilkan jarak dari tengah
            offset = cx - frame_w // 2
            cv.putText(
                frame,
                f"Offset: {offset}",
                (x, y + h + 60),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
    
    # Kirim data ke Arduino
    try:
        ser.write(f"{cx_to_send}\n".encode())
        print(f"Sent to Arduino: {cx_to_send}")
    except Exception as e:
        print(f"Serial error: {e}")
    
    # Tampilkan
    cv.imshow("Camera", frame)
    cv.imshow("Mask", mask)
    cv.imshow("Result", res)

    key = cv.waitKey(1) & 0xff
    if key == ord('q'):
        break
    elif key == ord('s'):  # Simpan screenshot
        cv.imwrite("screenshot.jpg", frame)
        print("Screenshot saved!")

# Cleanup
cam.release()
cv.destroyAllWindows()
ser.close()