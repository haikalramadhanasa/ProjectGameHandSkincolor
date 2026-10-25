import cv2
import numpy as np

# Buka Kamera
cap = cv2.VideoCapture(0)

# --- VARIABEL GAME MECHANICS ---
game_state = "CHARGE"  # Pilihan: "CHARGE", "THROWING", "BOUNCING", "STOPPED"
charge_power = 0       # Kekuatan lemparan (0 - 100)
aim_angle_offset = 0   # Pergeseran sudut arah (trajectory) dari kotak kedua (-50 sampai 50)

ball_pos = [0, 0]      # Posisi bola [x, y]
ball_vel = [0.0, 0.0]  # Kecepatan bola [vx, vy]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # --- 1. Cotnroller bxoes ---
    box_size = 120
    
    # Kotak 1
    box1_x1 = int(w * 0.25) - (box_size // 2)
    box1_y1 = int(h * 0.5) - (box_size // 2)
    box1_x2 = box1_x1 + box_size
    box1_y2 = box1_y1 + box_size

    # Kotak 2
    box2_x1 = int(w * 0.75) - (box_size // 2)
    box2_y1 = int(h * 0.5) - (box_size // 2)
    box2_x2 = box2_x1 + box_size
    box2_y2 = box2_y1 + box_size

    roi1 = frame[box1_y1:box1_y2, box1_x1:box1_x2]
    roi2 = frame[box2_y1:box2_y2, box2_x1:box2_x2]

    # Helper function untuk mendeteksi struk putih di dalam ROI
    def detect_white_struk(roi_img):
        if roi_img.size == 0:
            return False
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        lower_marker = np.array([0, 0, 180], dtype=np.uint8)
        upper_marker = np.array([180, 50, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_marker, upper_marker)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.countNonZero(mask) > 400

    is_throw_triggered = detect_white_struk(roi1)
    
    # Deteksi posisi tangan di Kotak 2 (Kanan) untuk mengatur arah sudut lempar
    has_aim_control = False
    if roi2.size != 0:
        hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
        # Deteksi tangan (kulit atau putih) di kotak 2
        lower_skin = np.array([0, 20, 50], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask_skin = cv2.inRange(hsv2, lower_skin, upper_skin)
        
        contours, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_c) > 1000:
                M = cv2.moments(max_c)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    # Tentukan offset sudut berdasarkan posisi X tangan di dalam kotak 2
                    # Jika tangan di kiri kotak 2 -> arah kiri, di kanan kotak 2 -> arah kanan
                    relative_x = cX - (box_size / 2)
                    aim_angle_offset = int(relative_x * 0.8) # Skala kepekaan arah

    # --- 2. State play
    if game_state == "CHARGE":
        if not is_throw_triggered:
            charge_power += 1.2
            if charge_power >= 100:
                charge_power = 100
        else:
            if charge_power > 10:
                game_state = "THROWING"
                ball_pos = [w + 100, h - 100]
                # Kecepatan awal dipengaruhi power DAN sudut aim_angle_offset dari kotak kedua
                vx = int((charge_power * 0.22) + (aim_angle_offset * 0.15))
                vy = int(-charge_power * 0.35)
                ball_vel = [vx, vy]

    elif game_state == "THROWING":
        # Gerak Parabola di udara
        ball_pos[0] += ball_vel[0]
        ball_pos[1] += ball_vel[1]
        ball_vel[1] += 1.2  # Gravitasi

        # Jika bola membentur lantai bawah
        floor_limit = h - 60
        if ball_pos[1] >= floor_limit:
            ball_pos[1] = floor_limit
            # Pantulan (Bounce) - kurangi kecepatan vertikal dan balik arah
            ball_vel[1] = -abs(ball_vel[1]) * 0.5 
            # Gesekkan lantai memperlambat kecepatan horizontal
            ball_vel[0] *= 0.85
            
            if abs(ball_vel[1]) < 2.0:
                game_state = "STOPPED"
                ball_vel = [0, 0]

    # --- 3. Visualization
    # Kotak 1 (Kiri)
    cv2.rectangle(frame, (box1_x1, box1_y1), (box1_x2, box1_y2), (0, 255, 255), 2)
    cv2.putText(frame, "1. CHARGE / THROW", (box1_x1 - 10, box1_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Kotak 2 (Kanan)
    cv2.rectangle(frame, (box2_x1, box2_y1), (box2_x2, box2_y2), (255, 0, 255), 2)
    cv2.putText(frame, "2. KONTROL ARAH", (box2_x1 - 10, box2_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    # --- 4. Canvas gameplay----
    game_canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    game_canvas[0:h, 0:w] = frame
    cv2.line(game_canvas, (w, 0), (w, h), (100, 100, 100), 2)
    
    # Background Area Game
    cv2.rectangle(game_canvas, (w + 20, 20), (w * 2 - 20, h - 20), (30, 30, 30), -1)
    
    # Garis Lantai / Tanah Game
    cv2.line(game_canvas, (w + 20, h - 60), (w * 2 - 20, h - 60), (100, 100, 100), 2)

    # Judul Game
    cv2.putText(game_canvas, "GAME: AIM & BOUNCE (TEKAN 'R' UNTUK RESET)", (w + 30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if game_state == "CHARGE":
        cv2.putText(game_canvas, f"STATUS: CHARGING (Arah Offset: {aim_angle_offset})", (w + 30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Power Bar
        bar_x, bar_y, bar_w, bar_h = w + 40, 130, 280, 25
        cv2.rectangle(game_canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        filled_w = int((charge_power / 100) * bar_w)
        cv2.rectangle(game_canvas, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)
        cv2.rectangle(game_canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.putText(game_canvas, f"POWER: {int(charge_power)}%", (bar_x, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Prediksi Trajectory dengan memperhitungkan aim_angle_offset dari Kotak 2
        start_traj = (w + 100, h - 100)
        end_traj_x = int(start_traj[0] + (charge_power * 2.2) + aim_angle_offset)
        end_traj_y = int(start_traj[1] - (charge_power * 1.8))
        cv2.line(game_canvas, start_traj, (end_traj_x, end_traj_y), (255, 165, 0), 2, cv2.LINE_AA)
        cv2.putText(game_canvas, "Arah Trajectory", (start_traj[0], start_traj[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        cv2.putText(game_canvas, "-> Struk Putih di Kotak Kiri untuk Lempar!", (w + 30, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    elif game_state in ["THROWING", "STOPPED"]:
        status_str = "BOLA MELUNCUR & MEMANTUL!" if game_state == "THROWING" else "BOLA BERHENTI (Tekan 'R' untuk Reset)"
        cv2.putText(game_canvas, f"STATUS: {status_str}", (w + 30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
        # Gambar Bola
        cv2.circle(game_canvas, (int(ball_pos[0]), int(ball_pos[1])), 15, (0, 165, 255), -1)
        cv2.circle(game_canvas, (int(ball_pos[0]), int(ball_pos[1])), 15, (255, 255, 255), 2)

    # Tampilkan Jendela Utama
    cv2.imshow("Basketball Aim & Bounce Game", game_canvas)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Tombol Hotkey 'r' atau 'R' untuk mereset permainan secara manual
    elif key == ord('r') or key == ord('R'):
        game_state = "CHARGE"
        charge_power = 0
        aim_angle_offset = 0

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
