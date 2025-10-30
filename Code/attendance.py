# attendance_gui.py - GIAO DIỆN CÓ NÚT CHECK-IN / CHECK-OUT
import cv2
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import csv
from datetime import datetime, date
import threading

# ================================
# CẤU HÌNH
# ================================
MODEL_PATH = '../Model'
XML_PATH = './'
LOG_FILE = 'attendance_log.csv'

# Kiểm tra file
for path, name in [(XML_PATH, 'haarcascade_frontalface_default.xml'),
                   (MODEL_PATH, 'face_model.yml'),
                   (MODEL_PATH, 'label_map.txt')]:
    file = os.path.join(path, name)
    if not os.path.exists(file):
        messagebox.showerror("Lỗi", f"Không tìm thấy: {file}")
        exit()

# Load model
cascade = cv2.CascadeClassifier(os.path.join(XML_PATH, 'haarcascade_frontalface_default.xml'))
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(os.path.join(MODEL_PATH, 'face_model.yml'))

# Load tên
id_to_name = {}
with open(os.path.join(MODEL_PATH, 'label_map.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        idx, name = line.strip().split(':', 1)
        id_to_name[int(idx)] = name

# Tạo file CSV
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'date', 'checkin', 'checkout'])

# ================================
# GIAO DIỆN TKINTER
# ================================
class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CHẤM CÔNG - NHẬN DIỆN KHUÔN MẶT")
        self.root.geometry("900x600")
        self.root.configure(bg="#f0f0f0")

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.current_name = "Chưa nhận diện"
        self.current_confidence = 100
        self.last_seen = {}

        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        # --- Camera Frame ---
        self.cam_label = tk.Label(self.root, bg="black")
        self.cam_label.pack(pady=10)

        # --- Info Frame ---
        info_frame = tk.Frame(self.root, bg="#f0f0f0")
        info_frame.pack(pady=10)

        tk.Label(info_frame, text="Tên:", font=("Arial", 14), bg="#f0f0f0").grid(row=0, column=0, padx=5)
        self.name_label = tk.Label(info_frame, text=self.current_name, font=("Arial", 14, "bold"), fg="blue", bg="#f0f0f0")
        self.name_label.grid(row=0, column=1, padx=10)

        tk.Label(info_frame, text="Giờ:", font=("Arial", 14), bg="#f0f0f0").grid(row=1, column=0, padx=5)
        self.time_label = tk.Label(info_frame, text="", font=("Arial", 14), bg="#f0f0f0")
        self.time_label.grid(row=1, column=1, padx=10)

        # --- Nút bấm ---
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=15)

        self.checkin_btn = tk.Button(
            btn_frame, text="CHECK-IN", command=self.manual_checkin,
            bg="#28a745", fg="white", font=("Arial", 12, "bold"), width=12, height=2
        )
        self.checkin_btn.grid(row=0, column=0, padx=10)

        self.checkout_btn = tk.Button(
            btn_frame, text="CHECK-OUT", command=self.manual_checkout,
            bg="#dc3545", fg="white", font=("Arial", 12, "bold"), width=12, height=2
        )
        self.checkout_btn.grid(row=0, column=1, padx=10)

        tk.Button(
            btn_frame, text="Thoát", command=self.root.quit,
            bg="#6c757d", fg="white", font=("Arial", 12, "bold"), width=12, height=2
        ).grid(row=0, column=2, padx=10)

        # --- Trạng thái ---
        self.status_label = tk.Label(self.root, text="Sẵn sàng", font=("Arial", 12), fg="green", bg="#f0f0f0")
        self.status_label.pack(pady=5)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize và detect
            small = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.2, 3, minSize=(50, 50))

            name = "Không thấy ai"
            confidence = 100

            if len(faces) > 0:
                x, y, w, h = faces[0]
                scale_x = 640 / 320
                scale_y = 480 / 240
                x2, y2, w2, h2 = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
                roi = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y2:y2+h2, x2:x2+w2], (100, 100))
                label, confidence = recognizer.predict(roi)
                if confidence < 90:
                    name = id_to_name.get(label, "Ai đó?")
                    self.current_name = name
                    self.current_confidence = confidence
                else:
                    name = "Không nhận diện"

                # Vẽ khung
                cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Cập nhật giao diện
            self.name_label.config(text=name)
            self.time_label.config(text=time.strftime("%H:%M:%S"))

            # Hiển thị camera
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def log_attendance(self, action):
        name = self.current_name
        if name in ["Không thấy ai", "Không nhận diện", "Ai đó?"]:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhận diện khuôn mặt trước!")
            return

        today = date.today().isoformat()
        now = datetime.now().strftime("%H:%M:%S")

        rows = []
        found = False
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                header = rows[0]
                data = rows[1:]

            for row in data:
                if len(row) >= 2 and row[0] == name and row[1] == today:
                    if action == "in" and not row[2]:
                        row[2] = now
                        messagebox.showinfo("Thành công", f"CHECK-IN: {name} - {now}")
                    elif action == "out" and row[2] and not row[3]:
                        row[3] = now
                        messagebox.showinfo("Thành công", f"CHECK-OUT: {name} - {now}")
                    else:
                        messagebox.showinfo("Thông báo", f"Đã {action == 'in' and 'Check-in' or 'Check-out'} rồi!")
                    found = True
                    break

        if not found and action == "in":
            data.append([name, today, now, ""])
            messagebox.showinfo("Thành công", f"CHECK-IN: {name} - {now}")

        # Ghi lại
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

        self.status_label.config(text=f"{action == 'in' and 'Check-in' or 'Check-out'} thành công!", fg="green")

    def manual_checkin(self):
        threading.Thread(target=self.log_attendance, args=("in",), daemon=True).start()

    def manual_checkout(self):
        threading.Thread(target=self.log_attendance, args=("out",), daemon=True).start()

    def __del__(self):
        self.cap.release()

# ================================
# CHẠY ỨNG DỤNG
# ================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()