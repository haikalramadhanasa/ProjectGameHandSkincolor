import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# Buka Kamera
cap = cv2.VideoCapture(0)

# Batas kecerahan
Threshold = 130

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka webcam.")
        break

    # Flip agar jadi cermin
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Konversi BGR ke RGB & Ubah format ke Image mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # deteksi tangan
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Ambil koordinat pusat 
            cx = int(hand_landmarks[9].x * w)
            cy = int(hand_landmarks[9].y * h)

            # Buat area ROI 
            box_size = 15
            y1, y2 = max(0, cy - box_size), min(h, cy + box_size)
            x1, x2 = max(0, cx - box_size), min(w, cx + box_size)

            roi = frame[y1:y2, x1:x2]

            if roi.size != 0:
                # Konversi ROI ke HSV
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mean_brightness = np.mean(hsv_roi[:, :, 2]) # Channel V (Value)

                # Klasifikasi berdasarkan sampel warna/kecerahan
                if mean_brightness > Threshold:
                    label = f"TELAPAK (Bright: {int(mean_brightness)})"
                    color = (0, 255, 0)
                else:
                    label = f"PUNGGUNG (Bright: {int(mean_brightness)})"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (cx - 60, cy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Window
    cv2.imshow("Game", frame)
    
    # Q untuk stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
