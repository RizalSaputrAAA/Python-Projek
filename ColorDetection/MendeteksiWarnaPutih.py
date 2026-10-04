import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

# Parameter untuk perhitungan jarak
# Nilai ini perlu dikalibrasi sesuai dengan setup kamera dan objek Anda
KNOWN_WIDTH = 3.5  # Lebar objek dalam cm (sesuaikan dengan objek nyata)
FOCAL_LENGTH = 600  # Nilai focal length dalam pixel (perlu dikalibrasi)

# Fungsi untuk menghitung jarak
def distance_to_camera(known_width, focal_length, per_width):
    if per_width == 0:  
        return 0
    return (known_width * focal_length) / per_width

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Range putih
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv.inRange(hsv, lower_blue, upper_blue)
    
    # Operasi morfologi untuk membersihkan noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    res = cv.bitwise_and(frame, frame, mask=mask)

    # Cari kontur
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv.boundingRect(cnt)
            
            distance = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, w)
            
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            label = f"WHITE: {distance:.1f} cm"
            cv.putText(frame, label, (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
            center_x = x + w//2
            center_y = y + h//2
            cv.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
            
            info_text = f"Width: {w}px, Area: {area:.0f}"
            cv.putText(frame, info_text, (x, y+h+20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Tampilkan informasi kalibrasi di frame
    cv.putText(frame, f"Focal: {FOCAL_LENGTH}, Known Width: {KNOWN_WIDTH}cm", 
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.imshow("frame", frame)
    cv.imshow("mask", mask)
    cv.imshow("res", res)

    key = cv.waitKey(1) & 0xFF
    
    # Opsi untuk kalibrasi real-time (opsional)
    if key == ord('c'):
        # Simpan nilai focal length saat ini
        print(f"Current focal length: {FOCAL_LENGTH}")
    
    if key == ord('+'):
        FOCAL_LENGTH += 10
        print(f"Focal length increased to: {FOCAL_LENGTH}")
    
    if key == ord('-'):
        FOCAL_LENGTH -= 10
        print(f"Focal length decreased to: {FOCAL_LENGTH}")
    
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()