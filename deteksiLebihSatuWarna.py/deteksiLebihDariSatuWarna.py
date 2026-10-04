import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

# Parameter untuk perhitungan jarak
KNOWN_WIDTH_WHITE = 3.5  # Lebar objek putih dalam cm
KNOWN_WIDTH_BLUE = 3.5   # Lebar objek biru dalam cm (sesuaikan jika berbeda)
FOCAL_LENGTH = 600       # Nilai focal length dalam pixel (perlu dikalibrasi)

# Fungsi untuk menghitung jarak
def distance_to_camera(known_width, focal_length, per_width):
    if per_width == 0:
        return 0
    return (known_width * focal_length) / per_width

# Range warna HSV
# Untuk warna putih
lower_white = np.array([0, 0, 200])
upper_white = np.array([179, 40, 255])

# Untuk warna biru - perlu disesuaikan dengan kondisi cahaya
# Range 1 untuk biru tua/terang
lower_blue1 = np.array([90, 50, 50])
upper_blue1 = np.array([130, 255, 255])

# Range 2 untuk biru muda (opsional, untuk mencakup lebih banyak nuansa biru)
lower_blue2 = np.array([100, 50, 50])
upper_blue2 = np.array([140, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Blur untuk mengurangi noise
    blurred = cv.GaussianBlur(hsv, (5, 5), 0)

    # Mask untuk warna putih
    mask_white = cv.inRange(blurred, lower_white, upper_white)
    
    # Mask untuk warna biru (gabungkan dua range)
    mask_blue1 = cv.inRange(blurred, lower_blue1, upper_blue1)
    mask_blue2 = cv.inRange(blurred, lower_blue2, upper_blue2)
    mask_blue = cv.bitwise_or(mask_blue1, mask_blue2)
    
    # Operasi morfologi untuk membersihkan noise
    kernel = np.ones((5,5), np.uint8)
    
    # Untuk warna putih
    mask_white = cv.morphologyEx(mask_white, cv.MORPH_OPEN, kernel)
    mask_white = cv.morphologyEx(mask_white, cv.MORPH_CLOSE, kernel)
    
    # Untuk warna biru
    mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_OPEN, kernel)
    mask_blue = cv.morphologyEx(mask_blue, cv.MORPH_CLOSE, kernel)
    
    # Gabungkan mask untuk menampilkan hasil deteksi semua warna
    mask_combined = cv.bitwise_or(mask_white, mask_blue)
    
    # Hasil deteksi warna
    res_white = cv.bitwise_and(frame, frame, mask=mask_white)
    res_blue = cv.bitwise_and(frame, frame, mask=mask_blue)
    res_combined = cv.bitwise_or(res_white, res_blue)

    # Cari kontur untuk warna putih
    contours_white, _ = cv.findContours(mask_white, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Cari kontur untuk warna biru
    contours_blue, _ = cv.findContours(mask_blue, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Deteksi objek putih
    for cnt in contours_white:
        area = cv.contourArea(cnt)
        if area > 500:  # Filter noise, hanya area besar
            x, y, w, h = cv.boundingRect(cnt)
            
            # Hitung jarak
            distance = distance_to_camera(KNOWN_WIDTH_WHITE, FOCAL_LENGTH, w)
            
            # Gambar bounding box (hijau untuk putih)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Tampilkan informasi
            label = f"WHITE: {distance:.1f} cm"
            cv.putText(frame, label, (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Tampilkan titik tengah
            center_x = x + w//2
            center_y = y + h//2
            cv.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # Deteksi objek biru
    for cnt in contours_blue:
        area = cv.contourArea(cnt)
        if area > 500:  # Filter noise, hanya area besar
            x, y, w, h = cv.boundingRect(cnt)
            
            # Hitung jarak
            distance = distance_to_camera(KNOWN_WIDTH_BLUE, FOCAL_LENGTH, w)
            
            # Gambar bounding box (merah untuk biru)
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Tampilkan informasi
            label = f"BLUE: {distance:.1f} cm"
            cv.putText(frame, label, (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Tampilkan titik tengah
            center_x = x + w//2
            center_y = y + h//2
            cv.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

    # Tampilkan informasi kalibrasi di frame
    cv.putText(frame, f"Focal: {FOCAL_LENGTH}", 
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv.putText(frame, f"White Width: {KNOWN_WIDTH_WHITE}cm, Blue Width: {KNOWN_WIDTH_BLUE}cm", 
               (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Tampilkan semua jendela
    cv.imshow("Frame - Deteksi Warna", frame)
    cv.imshow("Mask Putih", mask_white)
    cv.imshow("Mask Biru", mask_blue)
    cv.imshow("Mask Gabungan", mask_combined)
    cv.imshow("Hasil Deteksi", res_combined)

    key = cv.waitKey(1) & 0xFF
    
    # Opsi untuk kalibrasi real-time
    if key == ord('w'):  # Ubah lebar objek putih
        KNOWN_WIDTH_WHITE = float(input("Masukkan lebar objek putih (cm): "))
        print(f"Lebar objek putih diubah menjadi: {KNOWN_WIDTH_WHITE} cm")
    
    if key == ord('b'):  # Ubah lebar objek biru
        KNOWN_WIDTH_BLUE = float(input("Masukkan lebar objek biru (cm): "))
        print(f"Lebar objek biru diubah menjadi: {KNOWN_WIDTH_BLUE} cm")
    
    if key == ord('+'):  # Tingkatkan focal length
        FOCAL_LENGTH += 10
        print(f"Focal length meningkat menjadi: {FOCAL_LENGTH}")
    
    if key == ord('-'):  # Kurangi focal length
        FOCAL_LENGTH -= 10
        print(f"Focal length menurun menjadi: {FOCAL_LENGTH}")
    
    if key == ord('c'):  # Tampilkan nilai saat ini
        print(f"Focal Length: {FOCAL_LENGTH}")
        print(f"White Width: {KNOWN_WIDTH_WHITE} cm")
        print(f"Blue Width: {KNOWN_WIDTH_BLUE} cm")
    
    if key == ord('h'):  # Tampilkan bantuan
        print("\n=== Bantuan ===")
        print("w: Ubah lebar objek putih")
        print("b: Ubah lebar objek biru")
        print("+: Tingkatkan focal length")
        print("-: Kurangi focal length")
        print("c: Tampilkan nilai kalibrasi")
        print("h: Tampilkan bantuan")
        print("q: Keluar")
    
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()