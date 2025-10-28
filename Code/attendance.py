import cv2
import os
import time
import winsound  # ✅ Thêm âm thanh trên Windows

# ================================
# CẤU HÌNH
# ================================
MODEL_PATH = '../Model'
XML_PATH = './'

# Kiểm tra file
for path, name in [(XML_PATH, 'haarcascade_frontalface_default.xml'),
                   (MODEL_PATH, 'face_model.yml'),
                   (MODEL_PATH, 'label_map.txt')]:
    file = os.path.join(path, name)
    if not os.path.exists(file):
        print(f"Không tìm thấy: {file}")
        exit()

# Load model
cascade = cv2.CascadeClassifier(os.path.join(XML_PATH, 'haarcascade_frontalface_default.xml'))
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(MODEL_PATH, 'face_model.yml'))

# Load tên (UTF-8)
id_to_name = {}
with open(os.path.join(MODEL_PATH, 'label_map.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        idx, name = line.strip().split(':', 1)
        id_to_name[int(idx)] = name

# Mở camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
cap.set(5, 10)  # GIỚI HẠN 10 FPS → SIÊU NHẸ

# Biến chấm công
last_seen = ""
last_time = 0
COOLDOWN = 3

print("CHẤM CÔNG - ĐANG CHẠY... (Q để thoát)")

# ================================
# HÀM VẼ CHỮ TIẾNG VIỆT AN TOÀN
# ================================
def put_vn_text(img, text, pos, color, size=0.8, thick=2):
    text = "".join(c for c in text if ord(c) < 0x1EF9 or c in "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
    return img

# ================================
# VÒNG LẶP SIÊU NHẸ
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray, 1.2, 3, minSize=(50, 50))
    
    name = "Khong thay ai"
    now = time.time()

    if len(faces) > 0:
        x, y, w, h = faces[0]
        scale_x = 640 / 320
        scale_y = 480 / 240
        x, y, w, h = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)

        roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = roi_gray[y:y+h, x:x+w]
        if roi.size > 0:
            roi = cv2.resize(roi, (100, 100))
            label, confidence = recognizer.predict(roi)
            if confidence < 90:
                name = id_to_name.get(label, "Ai do?")
                if name != last_seen or (now - last_time) > COOLDOWN:
                    log_time = time.strftime("%H:%M:%S")
                    print(f"CHAM CONG: {name} - {log_time}")
                    with open("chamcong.log", "a", encoding="utf-8") as f:
                        f.write(f"{name},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    last_seen = name
                    last_time = now

                    # ✅ Phát tiếng kêu thành công
                    winsound.Beep(1000, 200)  # 1000Hz trong 200ms

            else:
                name = "Khong nhan dien"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        frame = put_vn_text(frame, name, (x, y-10), (0, 255, 0), 0.7, 2)

    frame = put_vn_text(frame, f"Trang thai: {name}", (10, 30), (0, 0, 255), 0.9, 2)

    cv2.imshow("CHAM CONG", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
