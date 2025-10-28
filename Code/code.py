import tkinter as tk
from tkinter import messagebox
import cv2
import os
import numpy as np

# ✅ Giảm cảnh báo GStreamer khi mở webcam
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ✅ Đường dẫn
XML_PATH = './'
DATASET_PATH = '../Dataset'
MODEL_PATH = '../Model'

# ✅ Tạo thư mục nếu chưa có
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)


class RegisterFaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 ĐĂNG KÝ KHUÔN MẶT NHÂN VIÊN")
        self.root.geometry("500x400")
        self.root.resizable(False, False)

        # --- Load bộ phát hiện khuôn mặt ---
        self.detector = self.load_cascade()
        self.create_widgets()

    def load_cascade(self):
        """Nạp file Haar Cascade"""
        xml_path = os.path.join(XML_PATH, 'haarcascade_frontalface_default.xml')
        if os.path.exists(xml_path):
            detector = cv2.CascadeClassifier(xml_path)
            if not detector.empty():
                print("✅ Đã tải bộ phát hiện khuôn mặt!")
                return detector
        print("❌ Không tìm thấy file haarcascade_frontalface_default.xml!")
        return None

    def create_widgets(self):
        """Giao diện chính"""
        tk.Label(
            self.root, text="📸 ĐĂNG KÝ KHUÔN MẶT NHÂN VIÊN",
            font=("Arial", 16, "bold"), fg="blue"
        ).pack(pady=15)

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        tk.Label(frame, text="Họ tên:", font=("Arial", 12)).grid(row=0, column=0, padx=5, pady=5)
        self.name = tk.Entry(frame, width=30, font=("Arial", 12))
        self.name.grid(row=0, column=1, padx=5, pady=5)

        tk.Button(
            self.root, text="🎥 BẮT ĐẦU CHỤP 30 ẢNH",
            command=self.on_register, bg="#28a745", fg="white",
            font=("Arial", 12, "bold"), width=25, height=2
        ).pack(pady=15)

        tk.Label(
            self.root,
            text="Nhấn 'Q' để thoát khi đang chụp.",
            fg="gray", font=("Arial", 10)
        ).pack(side="bottom", pady=10)

    def on_register(self):
        """Xử lý khi nhấn nút đăng ký"""
        name = self.name.get().strip()
        if not name:
            messagebox.showwarning("⚠️ Cảnh báo", "Vui lòng nhập Họ tên!")
            return

        # --- Ghi nhãn vào file label_map.txt ---
        label_map_path = os.path.join(MODEL_PATH, "label_map.txt")
        next_index = 0
        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                existing_names = [line.split(":", 1)[1] for line in lines if ":" in line]
                if name in existing_names:
                    messagebox.showinfo("ℹ️ Thông báo", f"Tên {name} đã có trong dữ liệu!")
                    return
                next_index = len(lines)

        with open(label_map_path, "a", encoding="utf-8") as f:
            f.write(f"{next_index}:{name}\n")

        self.capture_face(name)
        self.train_and_save_model()

    def capture_face(self, name):
        """Chụp và lưu 30 ảnh khuôn mặt theo tên thật"""
        if self.detector is None or self.detector.empty():
            messagebox.showerror("❌ Lỗi", "Không tải được file XML Haar Cascade!")
            return

        emp_folder = os.path.join(DATASET_PATH, name)
        os.makedirs(emp_folder, exist_ok=True)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            messagebox.showerror("❌ Lỗi", "Không thể mở camera!")
            return

        count = 0
        print(f"📸 Đang chụp ảnh cho {name}... (Nhấn Q để dừng)")

        while count < 30:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                cv2.imwrite(f"{emp_folder}/{count}.jpg", face_roi)
                count += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({count}/30)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "👀 Hãy nhìn thẳng vào camera!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("🚀 CHỤP KHUÔN MẶT", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Hoàn tất chụp 30 ảnh cho {name}!")

    def train_and_save_model(self):
        """Train LBPH model và lưu"""
        print("⚡ Đang train model LBPH...")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces = []
        labels = []

        # Load label_map
        label_map_path = os.path.join(MODEL_PATH, "label_map.txt")
        id_to_name = {}
        with open(label_map_path, "r", encoding="utf-8") as f:
            for line in f:
                idx, name = line.strip().split(":")
                id_to_name[int(idx)] = name

        # Load dataset theo tên thật
        for idx, name in id_to_name.items():
            folder = os.path.join(DATASET_PATH, name)
            if not os.path.isdir(folder):
                continue
            for img_name in os.listdir(folder):
                img_path = os.path.join(folder, img_name)
                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    faces.append(gray)
                    labels.append(idx)

        if not faces:
            print("❌ Không có dữ liệu để train!")
            return

        recognizer.train(faces, np.array(labels))
        model_file = os.path.join(MODEL_PATH, "face_model.yml")
        recognizer.save(model_file)
        print(f"✅ Model đã lưu tại {model_file}")


# --- Chạy chương trình ---
if __name__ == "__main__":
    root = tk.Tk()
    app = RegisterFaceApp(root)
    root.mainloop()
