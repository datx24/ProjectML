import os
import cv2
import numpy as np
import sys

# ================================
# CẤU HÌNH ĐƯỜNG DẪN
# ================================
DATASET_PATH = "../Dataset"
MODEL_PATH = "../Model"
os.makedirs(MODEL_PATH, exist_ok=True)

# ================================
# BƯỚC 1: KIỂM TRA DỮ LIỆU
# ================================
if not os.path.exists(DATASET_PATH):
    print("❌ Không tìm thấy thư mục Dataset!")
    sys.exit()

subfolders = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
if len(subfolders) == 0:
    print("❌ Dataset trống, chưa có dữ liệu khuôn mặt nào!")
    sys.exit()

# ================================
# BƯỚC 2: ĐỌC ẢNH VÀ GÁN NHÃN
# ================================
faces = []
labels = []
id_to_name = {}

print("⚡ Đang tải dữ liệu khuôn mặt...")

for label_id, folder_name in enumerate(subfolders):
    folder_path = os.path.join(DATASET_PATH, folder_name)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(image_files) == 0:
        print(f"⚠️ Bỏ qua thư mục {folder_name}: không có ảnh hợp lệ.")
        continue

    id_to_name[label_id] = folder_name

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"⚠️ Không đọc được ảnh: {img_path}")
            continue
        faces.append(gray)
        labels.append(label_id)

print(f"✅ Đã nạp {len(faces)} ảnh từ {len(id_to_name)} người.")

# ================================
# BƯỚC 3: KIỂM TRA DỮ LIỆU TRƯỚC KHI TRAIN
# ================================
if len(faces) == 0:
    print("❌ Không có dữ liệu để train!")
    sys.exit()

# ================================
# BƯỚC 4: TRAIN MODEL LBPH
# ================================
print("🚀 Đang train model LBPH...")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# ================================
# BƯỚC 5: LƯU MODEL VÀ LABEL_MAP
# ================================
model_file = os.path.join(MODEL_PATH, "face_model.yml")
recognizer.save(model_file)

label_map_path = os.path.join(MODEL_PATH, "label_map.txt")
with open(label_map_path, "w", encoding="utf-8") as f:
    for idx, name in id_to_name.items():
        f.write(f"{idx}:{name}\n")

print("✅ Đã lưu model tại:", model_file)
print("✅ Đã lưu label map tại:", label_map_path)
print("🎉 Huấn luyện hoàn tất!")
