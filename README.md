# 🛸 UAV Human Tracking System — YOLO26 & ByteTrack / BoT-SORT

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26%20(NMS--Free)-00599C.svg?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![Trackers](https://img.shields.io/badge/Trackers-ByteTrack%20%7C%20BoT--SORT-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20OOP-orange.svg)]()
[![Release](https://img.shields.io/badge/Release-v1.1-blue.svg)]()

> **Đề tài NCKH:** Nhận diện & bám vết người thời gian thực cho thiết bị bay không người lái (UAV) sử dụng kiến trúc YOLO26 Native NMS-Free kết hợp ByteTrack & BoT-SORT.

---

## 📌 1. TỔNG QUAN HỆ THỐNG & ĐẶC TÍNH KỸ THUẬT

- **Kiến trúc Native NMS-Free:** Sử dụng One-to-One Head (loại bỏ tầng lọc NMS và phân phối DFL), đưa độ trễ hậu xử lý xuống $\le 0.35 - 1.0\text{ ms/frame}$, tốc độ toàn chuỗi đạt $\ge 80 - 170\text{ FPS}$ trên GPU.
- **Hợp nhất Dữ liệu Đa tầng Góc nhìn:**
  - *VisDrone-DET (3.739 ảnh sạch):* Đã mask đen 81.540 vùng nhiễu *ignored*, tối ưu dải độ cao 5–20m góc xiên $15^\circ - 45^\circ$.
  - *P-DESTRE 4K (75 video chuỗi):* Khắc phục triệt để điểm mù góc dốc thẳng đứng $45^\circ - 90^\circ$ (Bird's-Eye View).
- **Kiến trúc Vận hành Thực tế:**
  - *Ground Station (Hiện tại):* Drone truyền video thô về laptop trạm mặt đất qua sóng vô tuyến/Wi-Fi; GPU máy trạm chạy `yolo26n_merged` / `yolo26s_merged` + ByteTrack/BoT-SORT hiển thị Telemetry HUD hỗ trợ phi công lái thủ công bằng tay cầm RC.
  - *Onboard Edge (Định hướng mở rộng):* Nghiên cứu lượng tử hóa TensorRT (FP16/INT8) triển khai mô hình Nano `yolo26n_merged` lên máy tính nhúng trên drone.

---

## ✨ 2. CÓ GÌ MỚI Ở BẢN CẬP NHẬT v1.1 (WHAT'S NEW IN v1.1)

Bản cập nhật **v1.1** tập trung tối ưu hóa tốc độ thời gian thực, chuẩn hóa điểm truy cập dòng lệnh và nâng cấp chất lượng luồng video:

1. **Chuẩn hóa Single CLI Entrypoint (`track_drone.py`):**
   - Hợp nhất toàn bộ các phương thức thu nhận (tệp video, webcam laptop, luồng Wi-Fi RTSP drone, custom receiver) về duy nhất tệp điều phối `track_drone.py`.
   - Hỗ trợ nạp động class bộ thu mới ngoài repo qua cờ `--receiver-module`.
2. **Mô hình Mặc định Siêu Tốc `yolo26n_merged` (Nano 2.5M):**
   - Chuyển mô hình mặc định sang `yolo26n_merged` với độ trễ suy luận chỉ $3.5 - 7.5\text{ ms}$, đưa hệ thống đạt chuẩn $\ge 30 - 60\text{ FPS}$ mượt mà ngay cả trên các khung cảnh phức tạp.
3. **Cơ chế Trích xuất Thích ứng (Adaptive Event-Triggered Memory Bank):**
   - Tự động nhận diện tương tác gần / giao cắt ($\text{IoU} \ge 0.10$ hoặc khoảng cách tâm $< 0.85 \times w_{avg}$) để ép trích xuất $100\%$ mọi frame chống nhảy ID (ID Switch).
   - Với các mục tiêu ổn định: chỉ cập nhật template sau mỗi $3$ frames, cắt giảm $21.7\%$ chi phí tính toán CPU trên video đám đông $180$ người.
4. **Bộ Ghi Video H.264 P2 Visual Lossless & Cửa Sổ Xem Trực Tiếp 0ms:**
   - Bộ ghi tệp nén bằng `libx264` ở mức `CRF 18` kèm In-loop Deblocking Filter, giữ nguyên $100\%$ độ phân giải gốc của video đầu vào (Full HD / 4K).
   - Cửa sổ xem trước `--view` hiển thị trực tiếp từ RAM (Direct RAM View) với độ trễ $0\text{ ms}$, cắt giảm thêm $15\text{ ms}$ lãng phí CPU.

---

## 🛠️ 3. CÀI ĐẶT MÔI TRƯỜNG (INSTALLATION)

### 3.1. Yêu Cầu Tiên Quyết
- **Hệ điều hành:** Windows 10/11 hoặc Linux (Ubuntu 20.04/22.04).
- **Python:** Phiên bản **3.10 – 3.12** (khuyên dùng Python 3.11 hoặc 3.12).
- **Card đồ họa:** NVIDIA GPU hỗ trợ CUDA (khuyên dùng CUDA 12.1+).

### 3.2. Các Bước Cài Đặt

```powershell
# Bước 1: Clone repository và mở thư mục dự án:
git clone https://github.com/TheUn4given123jkbkjs/human-tracking-drone.git
cd human-tracking-drone

# Bước 2: Khởi tạo và kích hoạt môi trường ảo:
python -m venv .venv

# Trên Windows (PowerShell):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1

# Trên Linux / macOS:
source .venv/bin/activate

# Bước 3: Cài đặt PyTorch hỗ trợ GPU CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Bước 4: Cài đặt toàn bộ các thư viện phụ thuộc:
pip install -r requirements.txt
```

---

## 💻 4. HƯỚNG DẪN SỬ DỤNG (CLI REFERENCE)

Hệ thống điều phối toàn bộ luồng xử lý qua tệp thực thi duy nhất `track_drone.py`.

```powershell
# Kiểm tra danh sách mô hình hợp lệ có sẵn trên máy:
python track_drone.py --list-models
```

### 4.1. Cú Pháp Khởi Chạy Nhanh

**Cú pháp tổng quát:**
```powershell
python track_drone.py --source <nguồn_vào> [--model <model>] [--tracker <tracker>] [--view] [tùy_chọn]
```

**Các ví dụ phổ biến:**
```powershell
# 1. Chạy trên video flycam (Mặc định yolo26n_merged, xuất file H.264 siêu nét):
python track_drone.py --source data/gettyimages-1345327468-640_adpp.mp4

# 2. Chạy có cửa sổ theo dõi trực tiếp trên màn hình (Direct RAM View 0ms latency):
python track_drone.py --source data/gettyimages-1345327468-640_adpp.mp4 --view

# 3. Kết nối Webcam laptop (Tự động mở cửa sổ theo dõi):
python track_drone.py --source 0

# 4. Kết nối luồng Wi-Fi RTSP truyền trực tiếp từ Drone:
python track_drone.py --source rtsp://192.168.1.1:8554/live

# 5. Chạy mô hình trạm mặt đất yolo26s_merged kết hợp BoT-SORT bù rung lắc camera:
python track_drone.py --model yolo26s_merged --tracker botsort --source data/gettyimages-1345327468-640_adpp.mp4 --view
```

### 4.2. Bảng Tham Số Cốt Lõi (Core Parameters)

| Tham Số | Các Lựa Chọn | Mặc Định | Ý Nghĩa |
| :--- | :---: | :---: | :--- |
| **`--source`** | `File .mp4`, `Camera (0, 1)`, `URL (rtsp://, udp://)` | Video mẫu | Nguồn video đầu vào (tệp, webcam hoặc drone) |
| **`--view`** / **`--no-view`** | `--view`, `--no-view` | *Tự động* | Bật/tắt cửa sổ theo dõi trực tiếp |
| **`--model`** | `yolo26n_merged`, `yolo26s_merged`, `yolo26n_visdrone` | `yolo26n_merged` | Tên mô hình AI |
| **`--tracker`** | `bytetrack`, `botsort` | `bytetrack` | Thuật toán bám vết |
| **`--out`** | `Đường dẫn file .mp4 hoặc thư mục` | `runs/track/` | Đường dẫn lưu video H.264 đầu ra |

### 4.3. Bảng Tùy Chọn Nâng Cao (Advanced Options)

| Tham Số | Các Lựa Chọn | Mặc Định | Ý Nghĩa |
| :--- | :---: | :---: | :--- |
| **`--conf`** | `[0; 1]` | `0.20` | Ngưỡng tin cậy phát hiện |
| **`--iou`** | `[0; 1]` | `0.50` | Ngưỡng IoU liên kết vết |
| **`--step`** | `1`, `2` | `1` | Bước nhảy frame (1: 30 FPS, 2: 15 FPS) |
| **`--memory`** / **`--no-memory`** | `--memory`, `--no-memory` | `--memory` | Bật/tắt tầng Appearance Memory Bank |
| **`--adaptive-memory`** / **`--no-adaptive-memory`** | `--adaptive-memory`, `--no-adaptive-memory` | `--adaptive-memory` | Bật/tắt trích xuất thích ứng theo sự kiện |
| **`--feat-interval`** | `[1; 30]` | `3` | Chu kỳ làm mới template cho mục tiêu ổn định |
| **`--sim-thresh`** | `[0; 1]` | `0.60` | Ngưỡng Cosine Similarity để Re-ID |
| **`--max-lost-frames`** | `[1; 300]` | `60` | Số frame tối đa lưu mục tiêu mất dấu |
| **`--trail`** / **`--no-trail`** | `--trail`, `--no-trail` | `--no-trail` | Hiện/ẩn vệt quỹ đạo di chuyển |
| **`--stream-type`** | `auto`, `camera`, `rtsp`, `file` | `auto` | Giao thức/chế độ thu video |
| **`--record-fps`** | `Số thực > 0` | *Tự động* | Tốc độ FPS khi ghi video đầu ra |
| **`--receiver-module`** | `Đường dẫn file .py` | `None` | File Python chứa bộ thu tùy biến |
| **`--device`** | `0`, `cpu` | `0` | Thiết bị tính toán (GPU hoặc CPU) |
| **`--max-frames`** | `Số nguyên > 0` | `None` | Giới hạn số frame kiểm thử |
| **`--list-models`** | `--list-models` | `False` | Liệt kê các mô hình có sẵn và thoát |

> [!NOTE]
> **Mối quan hệ giữa `--source` và `--stream-type`:**
> - `--source` là **địa chỉ dữ liệu cụ thể** (bắt buộc khi muốn đổi nguồn: đường dẫn file, camera index `0`, hay URL RTSP).
> - `--stream-type` là **giao thức giải mã** (mặc định là `auto`). Cơ chế `auto` sẽ tự động nhận diện từ `--source`: nếu là số (`0`, `1`) tự chuyển sang `camera`, nếu bắt đầu bằng `rtsp://` tự chuyển sang `rtsp`, còn lại là `file`. Do đó, người dùng **chỉ cần truyền `--source`**, không cần gõ thêm `--stream-type`.
---

## 📊 5. KẾT QUẢ THỰC NGHIỆM NHẬN DIỆN (EVALUATION)

Chuẩn hóa đo lường cùng điều kiện trên Tesla T4 GPU (batch 16, imgsz 640), kiểm thử trên 440 ảnh VisDrone Val và 773 ảnh từ toàn bộ 15 video P-DESTRE Val:

| Mô Hình | Tham Số | Kích Thước | VisDrone Val ($mAP_{50}$) | VisDrone Recall | P-DESTRE Val 15vids ($mAP_{50}$) | P-DESTRE Recall | Độ Trễ Suy Luận (T4) | Đánh Giá Thực Nghiệm & Khoa Học |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`yolo26n_visdrone`** *(Baseline)* | 2.5M | $5.2\text{ MB}$ | **57.35%** | 56.38% | **47.61%** | 45.05% | **2.64 ms** | Tụt gần $-10\%$ $mAP$ ở góc dốc $45^\circ-90^\circ$ do thiếu dữ liệu đỉnh đầu |
| **`yolo26n_merged`** *(Nano)* | 2.5M | $5.2\text{ MB}$ | **58.30%** | 59.09% | **88.36%** | 86.52% | **3.15 ms** | Không quên tri thức cũ (+0.95%), đột phá góc dốc, tối ưu Edge Onboard |
| **`yolo26s_merged`** *(Small)* | **9.5M** | $19.4\text{ MB}$ | **61.47%** | **63.37%** | **66.80%** | 63.70% | **5.19 ms** | **Dẫn đầu toàn diện VisDrone (+4.12% mAP, +7% Recall), tối ưu trạm mặt đất** |

---

## 📈 6. GIAO DIỆN BẢNG THÔNG SỐ HUD (HEADS-UP DISPLAY) & THỊ GIÁC

Bảng đo lường Telemetry HUD hiển thị trực tiếp ở góc trên bên trái video với thiết kế thu nhỏ $220 \times 84\text{ px}$ (giảm hơn 60% diện tích, độ trong suốt 70%, chữ sắc nét, không che khuất mục tiêu):
* **MODEL | TRACKER:** Tên mô hình AI và thuật toán bám vết đang thực thi.
* **FPS | Inf:** Tốc độ khung hình tức thời (End-to-End Loop FPS) và độ trễ suy luận mạng nơ-ron (ms).
* **Active | Total IDs:** Số đối tượng đang bám vết đồng thời và tổng số ID duy nhất (được đánh số tuần tự từ `#1` cho từng video).
* **Re-ID:** Số lượt khôi phục danh tính thành công qua Memory Bank / Trạng thái TARGET LOCK.
* **BBox Labels:** Cỡ chữ gọn gàng (`font_scale=0.36`), độ dày nét viền hộp được tinh chỉnh mảnh gọn, badge nền ôm sát vừa vặn viền hộp.

### Quy Chuẩn Màu Sắc Bounding Box Ngữ Nghĩa:
Nhằm giúp người điều khiển và phi công dễ dàng nhận diện trạng thái của từng mục tiêu mà không bị rối mắt:
* 🟡 **Màu Vàng (Yellow):** Mục tiêu người mới phát hiện lần đầu trong tầm quan sát của camera.
* 🔵 **Màu Xanh biển (Sky Blue):** Mục tiêu từng bị che khuất hoặc mất dấu tạm thời và vừa được **Appearance Memory Bank** tái nhận diện (Re-ID) khôi phục lại danh tính thành công.
* 🔴 **Màu Đỏ (Red):** Mục tiêu đang được phi công hoặc trạm mặt đất khóa bám vết chính (**Locked Target**).

---

## 📜 7. TRÍCH DẪN KHOA HỌC

* **Ultralytics Framework (v8.4.x):** One-to-One Head Native NMS-Free Real-time Detection Architecture.
* **ByteTrack:** *Zhang et al.*, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
* **BoT-SORT:** *Aharon et al.*, "BoT-SORT: Robust Associations Multi-Pedestrian Tracking", arXiv:2206.14651, 2022.
* **P-DESTRE:** *Kumar et al.*, "P-DESTRE: Pedestrian Detection, Tracking, and Re-Identification from Aerial Devices", IEEE TPAMI, 2021.
* **VisDrone-DET:** *Zhu et al.*, "Vision Meets Drones: A Challenge", IEEE TPAMI, 2021.
