import cv2
import numpy as np

# Buka Kamera
cap = cv2.VideoCapture(0)

# Load image
bg_image = cv2.imread('backgroundgame.jpg')
fumo_sprite = cv2.imread('sparklefumoresized.png', cv2.IMREAD_UNCHANGED)
basket_sprite = cv2.imread('basket.png', cv2.IMREAD_UNCHANGED)

if fumo_sprite is not None:
    fumo_sprite = cv2.resize(fumo_sprite, (80, 80))
else:
    print("fumo file not found")
    
if basket_sprite is not None:
    basket_sprite = cv2.resize(basket_sprite, (90, 70))

def draw_transparent_image(background, foreground_img, x, y, angle):
    if foreground_img is None:
        cv2.circle(background, (int(x), int(y)), 25, (0, 165, 255), -1)
        return background

    h_fg, w_fg, _ = foreground_img.shape
    
    center = (w_fg // 2, h_fg // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated_fg = cv2.warpAffine(foreground_img, M, (w_fg, h_fg), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0, 0))
    
    x_start = int(x - (w_fg // 2))
    y_start = int(y - (h_fg // 2))
    
    # Batasi agar tidak keluar dari canvas sisi kanan (w sampai w*2)
    if y_start < 0 or x_start < w or y_start + h_fg > background.shape[0] or x_start + w_fg > background.shape[1]:
        return background
    
    roi = background[y_start:y_start+h_fg, x_start:x_start+w_fg]
    bgr_fg = rotated_fg[:, :, :3]
    alpha_fg = rotated_fg[:, :, 3] / 255.0 
    
    for c in range(3):
        roi[:, :, c] = (alpha_fg * bgr_fg[:, :, c] + (1.0 - alpha_fg) * roi[:, :, c])
        
    background[y_start:y_start+h_fg, x_start:x_start+w_fg] = roi
    return background

# --- VARIABEL GAME MECHANICS ---
game_state = "CHARGE"  # Pilihan: "Charge", "Throwing", "Stopped", "Victory"
charge_power = 0       # Kekuatan lemparan 
aim_angle_offset = 0   # Trajectory kotak kedua
fumo_angle = 0.0

ball_pos = [0, 0]      
ball_vel = [0.0, 0.0]  # Kecepatan bola [vx, vy]

score = 0
basket_x = 0
basket_y = 150
basket_speed = 3
basket_dir = 1 #(1 kanan -1 kiri)

#window resizeable
cv2.namedWindow("proyek game melempar fumo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("proyek game melempar fumo", 1280, 640)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    #posisi awal keranjang
    if basket_x == 0:
        basket_x = w + int(w / 2)
        
    #resize background
    if bg_image is not None:
        current_bg = cv2.resize(bg_image, (w, h))
    else:
        current_bg = np.zeros((h, w, 3), dtype=np.uint8) # Fallback jika gambar tidak ada

   # --- 1. DUA KOTAK KONTROL ---
    box_size = 120
    box1_x1 = int(w * 0.25) - (box_size // 2)
    box1_y1 = int(h * 0.5) - (box_size // 2)
    box1_x2 = box1_x1 + box_size
    box1_y2 = box1_y1 + box_size

    box2_x1 = int(w * 0.75) - (box_size // 2)
    box2_y1 = int(h * 0.5) - (box_size // 2)
    box2_x2 = box2_x1 + box_size
    box2_y2 = box2_y1 + box_size

    roi1 = frame[box1_y1:box1_y2, box1_x1:box1_x2]
    roi2 = frame[box2_y1:box2_y2, box2_x1:box2_x2]

    total_box_area = box_size * box_size # Total piksel dalam kotak (14400)
    yellow_ratio = 0.0
    skin_ratio = 0.0

    if roi1.size != 0:
        # Peningkatan kontras otomatis (CLAHE) agar stabil di ruangan redup
        img_yuv = cv2.cvtColor(roi1, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        roi_bright = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        hsv_roi = cv2.cvtColor(roi_bright, cv2.COLOR_BGR2HSV)
        kernel = np.ones((3, 3), np.uint8)

        # A. Hitung persentase Struk Putih Pure
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        yellow_ratio = yellow_pixels / total_box_area # Nilai dari 0.0 sampai 1.0

        # B. Hitung persentase Kulit Tangan
        lower_skin = np.array([0, 15, 40], dtype=np.uint8)
        upper_skin = np.array([25, 170, 240], dtype=np.uint8)
        mask_skin = cv2.inRange(hsv_roi, lower_skin, upper_skin)
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
        skin_pixels = cv2.countNonZero(mask_skin)
        skin_ratio = skin_pixels / total_box_area # Nilai dari 0.0 sampai 1.0 (0% - 100%)

    # Deteksi posisi tangan di Kotak 2 (Kanan) untuk arah sudut lempar
    if roi2.size != 0:
        hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
        lower_skin2 = np.array([0, 20, 50], dtype=np.uint8)
        upper_skin2 = np.array([20, 255, 255], dtype=np.uint8)
        mask_skin2 = cv2.inRange(hsv2, lower_skin2, upper_skin2)
        
        contours, _ = cv2.findContours(mask_skin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_c) > 1000:
                M = cv2.moments(max_c)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    relative_x = cX - (box_size / 2)
                    aim_angle_offset = int(relative_x * 0.8)
                    
    #moving keranjang
    if game_state != "VICTORY":
        basket_x += basket_speed * basket_dir
        if basket_x < w + 60 or basket_x > w * 2 - 60:
            basket_dir *= -1
        
    # --- 2. STATE MACHINE GAMEPLAY ---
    if game_state == "CHARGE":
        #release jika 30%
        if yellow_ratio >= 0.30:
            if charge_power > 10:
                game_state = "THROWING"
                ball_pos = [w + 100, h - 100]
                vx = int((charge_power * 0.22) + (aim_angle_offset * 0.15))
                vy = int(-charge_power * 0.35)
                ball_vel = [vx, vy]
        
        # Kondisi CHARGE NAIK: Jika tangan/kulit mengisi >= 50% area kotak
        elif skin_ratio >= 0.50:
            charge_power += 1.2
            if charge_power >= 100:
                charge_power = 100
        
        # Kondisi CHARGE TURUN: Jika tangan bergeser / keluar kotak (< 50% warna kulit)
        else:
            charge_power -= 1.5 # Power turun perlahan agar bisa dikontrol naik-turun
            if charge_power < 0:
                charge_power = 0

    elif game_state == "THROWING":
        ball_pos[0] += ball_vel[0]
        ball_pos[1] += ball_vel[1]
        ball_vel[1] += 1.2  # Gravitasi
        
        #-- rotasi fumo --
        
        fumo_angle -= ball_vel[0] * 2.5

        # --- BATASAN DAN PANTULAN DINDING (WINDOW BORDER BOUNCE) ---
        margin = 35  # Ukuran setengah dari sprite fumo agar pas di tepi
        game_left = w + margin
        game_right = (w * 2) - margin
        game_top = margin
        floor_limit = h - 60 - margin

        # 1. Pantulan Dinding Kiri (Batas garis pemisah kamera & game)
        if ball_pos[0] <= game_left:
            ball_pos[0] = game_left
            ball_vel[0] = -ball_vel[0] * 0.75  

        # 2. Pantulan Dinding Kanan 
        elif ball_pos[0] >= game_right:
            ball_pos[0] = game_right
            ball_vel[0] = -ball_vel[0] * 0.75

        # 3. Pantulan Atap 
        if ball_pos[1] <= game_top:
            ball_pos[1] = game_top
            ball_vel[1] = -ball_vel[1] * 0.75

     
        
        #cek fumo masuk keranjang
        if (basket_x - 35 < ball_pos[0] < basket_x + 35) and (basket_y - 25 < ball_pos[1] < basket_y + 25):
            score += 1
            if score >= 3:
                game_state = "VICTORY"
                score *= 0
                basket_speed = 3
            else:
                game_state = "CHARGE"
                charge_power = 0
                aim_angle_offset = 0
                basket_speed += 2

        floor_limit = h - 60
        if ball_pos[1] >= floor_limit:
            ball_pos[1] = floor_limit
            ball_vel[1] = -abs(ball_vel[1]) * 0.5 
            ball_vel[0] *= 0.85
            
            if abs(ball_vel[1]) < 2.0 and abs(ball_vel[0]) < 1.0:
                game_state = "STOPPED"
                ball_vel = [0, 0]

    # --- 3. VISUALISASI KOTAK DI FRAME KAMERA ---
    cv2.rectangle(frame, (box1_x1, box1_y1), (box1_x2, box1_y2), (0, 255, 255), 2)
    cv2.putText(frame, "Kontrol power", (box1_x1 - 10, box1_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.rectangle(frame, (box2_x1, box2_y1), (box2_x2, box2_y2), (255, 0, 255), 2)
    cv2.putText(frame, "Kontrol arah", (box2_x1 - 10, box2_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    # --- 4. CANVAS GAMEPLAY ----
    game_canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    #frame webcam
    game_canvas[0:h, 0:w] = frame
    #background frame
    game_canvas[0:h, w:w*2] = current_bg
    
    cv2.line(game_canvas, (w, 0), (w, h), (100, 100, 100), 2)
    cv2.line(game_canvas, (w + 20, h - 60), (w * 2 - 20, h - 60), (100, 100, 100), 2)
  
    if basket_sprite is not None:
        game_canvas = draw_transparent_image(game_canvas, basket_sprite, basket_x, basket_y, 0.0)
    else:
        # Fallback Keranjang
        cv2.rectangle(game_canvas, (int(basket_x - 30), int(basket_y - 15)), (int(basket_x + 30), int(basket_y + 15)), (0, 0, 255), 2)
        cv2.putText(game_canvas, "HOOP", (int(basket_x - 20), int(basket_y + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Tampilkan Skor di Layar Game
    cv2.putText(game_canvas, f"Skor: {score}/3 | Speed keranjang: {basket_speed}", (w + 35, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Judul Game
    cv2.putText(game_canvas, "Tekan 'R' untuk reset", (w + 30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    if game_state == "CHARGE":
        cv2.putText(game_canvas, f"STATUS: CHARGING (Offset: {aim_angle_offset})", (w + 30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Power Bar
        bar_x, bar_y, bar_w, bar_h = w + 40, 130, 280, 25
        cv2.rectangle(game_canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        filled_w = int((charge_power / 100) * bar_w)
        cv2.rectangle(game_canvas, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0), -1)
        cv2.rectangle(game_canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.putText(game_canvas, f"POWER: {int(charge_power)}%", (bar_x, bar_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # --- PREDIKSI TRAJECTORY PARABOLA MELENGKUNG ---
        start_x, start_y = w + 100, h - 100
        sim_vx = (charge_power * 0.22) + (aim_angle_offset * 0.15)
        sim_vy = -charge_power * 0.35
        sim_gravity = 1.2

        prev_x, prev_y = start_x, start_y
        for t in range(1, 25):
            curr_x = int(start_x + sim_vx * t)
            curr_y = int(start_y + sim_vy * t + 0.5 * sim_gravity * (t ** 2))
            
            if curr_x > w * 2 - 20 or curr_y > h - 60:
                break
                
            cv2.line(game_canvas, (prev_x, prev_y), (curr_x, curr_y), (255, 165, 0), 2, cv2.LINE_AA)
            prev_x, prev_y = curr_x, curr_y


    elif game_state in ["THROWING", "STOPPED"]:
        status_str = "fumo melayang" if game_state == "THROWING" else "Fumo mendarat (Press 'R' to reset)"
        cv2.putText(game_canvas, f"STATUS: {status_str}", (w + 30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
        # Gambar Fumo
        game_canvas = draw_transparent_image(game_canvas, fumo_sprite, ball_pos[0], ball_pos[1], fumo_angle)
    
    elif game_state == "VICTORY":
        cv2.rectangle(game_canvas, (w + 50, h // 2 - 80), (w * 2 - 50, h // 2 + 80), (20, 20, 20), -1)
        cv2.rectangle(game_canvas, (w + 50, h // 2 - 80), (w * 2 - 50, h // 2 + 80), (0, 255, 0), 3)
        cv2.putText(game_canvas, "game OVER", (w + 70, h // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(game_canvas, "tekan R untuk ulang", (w + 100, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Tampilkan Jendela Utama
    cv2.imshow("proyek game melempar fumo", game_canvas)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') or key == ord('R'):
        game_state = "CHARGE"
        charge_power = 0
        aim_angle_offset = 0

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
