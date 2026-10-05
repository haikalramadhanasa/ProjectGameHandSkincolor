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
        
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Ambil koordinat titik pusat telapak tangan
            center_x = int(hand_landmarks[9].x * w)
            center_y = int(hand_landmarks[9].y * h)
            
            # lingkaran merah di titik tengah tangan
            cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"Center: ({center_x}, {center_y})", (center_x + 15, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Window
    cv2.imshow("Tahap 1 - Deteksi Tangan (Python 3.13)", frame)
    # Q untuk stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()