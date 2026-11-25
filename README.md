# 🛸 UAV Human Tracking System — YOLO26 & ByteTrack / BoT-SORT

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26%20(NMS--Free)-00599C.svg?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![Trackers](https://img.shields.io/badge/Trackers-ByteTrack%20%7C%20BoT--SORT-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20OOP-orange.svg)]()
[![Release](https://img.shields.io/badge/Release-v1.0-blue.svg)]()

> **Đề tài NCKH:** Nhận diện & bám vết người thời gian thực cho thiết bị bay không người lái (UAV) sử dụng kiến trúc YOLO26 Native NMS-Free kết hợp ByteTrack & BoT-SORT.

---

## 📌 1. TỔNG QUAN HỆ THỐNG & ĐẶC TÍNH KỸ THUẬT

- **Kiến trúc Native NMS-Free:** Sử dụng One-to-One Head (loại bỏ tầng lọc NMS và phân phối DFL), đưa độ trễ hậu xử lý xuống $\le 0.35 - 1.0\text{ ms/frame}$, tốc độ toàn chuỗi đạt $\ge 80 - 170\text{ FPS}$ trên GPU.
- **Hợp nhất Dữ liệu Đa tầng Góc nhìn:**
  - *VisDrone-DET (3.739 ảnh sạch):* Đã mask đen 81.540 vùng nhiễu *ignored*, tối ưu dải độ cao 5–20m góc xiên $15^\circ - 45^\circ$.
  - *P-DESTRE 4K (75 video chuỗi):* Khắc phục triệt để điểm mù góc dốc thẳng đứng $45^\circ - 90^\circ$ (Bird's-Eye View).
- **Kiến trúc Vận hành Thực tế:**
  - *Ground Station (Hiện tại):* Drone truyền video thô về laptop trạm mặt đất qua sóng vô tuyến/Wi-Fi; GPU máy trạm chạy `yolo26s_merged` + ByteTrack/BoT-SORT hiển thị Telemetry HUD hỗ trợ phi công lái thủ công bằng tay cầm RC.
  - *Onboard Edge (Định hướng mở rộng):* Nghiên cứu lượng tử hóa TensorRT (FP16/INT8) triển khai mô hình Nano `yolo26n_merged` lên máy tính nhúng trên drone.

---

## 📊 2. KẾT QUẢ THỰC NGHIỆM ĐỐI CHỨNG

### 2.1. Hiệu Năng Nhận Diện Đối Tượng Đa Miền (Chuẩn Hóa Cùng Điều Kiện Trên 15 Video Val)

| Mô Hình | Tham Số | Kích Thước | VisDrone Val ($mAP_{50}$) | VisDrone Recall | P-DESTRE Val 15vids ($mAP_{50}$) | P-DESTRE Recall | Độ Trễ Suy Luận (T4) | Đánh Giá Thực Nghiệm & Khoa Học |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`yolo26n_visdrone`** *(Baseline)* | 2.5M | $5.2\text{ MB}$ | **57.35%** | 56.38% | **47.61%** | 45.05% | **2.64 ms** | Tụt gần $-10\%$ $mAP$ ở góc dốc $45^\circ-90^\circ$ do thiếu dữ liệu đỉnh đầu |
| **`yolo26n_merged`** *(Nano)* | 2.5M | $5.2\text{ MB}$ | **58.30%** | 59.09% | **88.36%** | 86.52% | **3.15 ms** | Không quên tri thức cũ (+0.95%), đột phá góc dốc, tối ưu Edge Onboard |
| **`yolo26s_merged`** *(Small)* | **9.5M** | $19.4\text{ MB}$ | **61.47%** | **63.37%** | **66.80%** | 63.70% | **5.19 ms** | **Dẫn đầu toàn diện VisDrone (+4.12% mAP, +7% Recall), tối ưu trạm mặt đất** |

*Ghi chú:* Bảng số liệu được đánh giá đồng bộ trên cùng môi trường (Tesla T4 GPU, batch 16, imgsz 640), kiểm thử trên 440 ảnh VisDrone Val và 773 ảnh từ toàn bộ 15 video P-DESTRE Val.

---

### 2.2. So Sánh & Thực Nghiệm Thuật Toán Bám Vết (ByteTrack vs BoT-SORT)

| Thuật Toán | Cơ Chế Liên Kết | Bù Rung Lắc Camera (GMC) | Đặc Điểm Vận Hành | Tình Huống Khuyến Nghị |
| :--- | :--- | :---: | :--- | :--- |
| **ByteTrack** | Kalman Filter + IoU Matching 2 tầng | Không | Tải tính toán siêu nhẹ ($< 2\text{ ms}$), FPS cực cao | **Mặc định trạm mặt đất & drone bay ổn định** |
| **BoT-SORT** | Kalman Filter + Camera Motion Compensation | **Có (Affine matrix)** | Bù trừ chuyển động nền, chống trôi dạt ID | **Khi drone chao liệng, quay vòng hoặc đổi hướng gắt** |

#### Bảng Đo Lường Thực Nghiệm Đối Chứng 4 Cấu Hình (Video Drone Chao Liệng — `test_drone_square_pdestre.mp4` — Full HD 1080p):

| Mô Hình AI | Thuật Toán Tracker | Tổng BBox Bắt Được | Số Unique IDs | BBox / Frame | Live Tracking FPS | Đánh Giá Thực Nghiệm & Hiện Tượng |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`yolo26n_merged`** *(2.5M)* | **ByteTrack** | 624 bboxes | 99 IDs | 7.61 | **~68 – 84 FPS** | Tốc độ cực cao, nhưng đứt gãy track do drone xoay góc |
| **`yolo26n_merged`** *(2.5M)* | **BoT-SORT** | **806 bboxes** | **66 IDs** | **9.83** | ~19 – 27 FPS | **Giữ track tốt nhất (-33% ID switches), bắt dính nhiều BBox nhất (+29%)** |
| **`yolo26s_merged`** *(9.5M)* | **ByteTrack** | 540 bboxes | 102 IDs | 6.59 | **~52 – 80 FPS** | Tốc độ trạm mặt đất siêu mượt, phù hợp luồng bay thẳng |
| **`yolo26s_merged`** *(9.5M)* | **BoT-SORT** | **736 bboxes** | **70 IDs** | **8.98** | ~18 – 27 FPS | **Giảm 31.4% ID switches nhờ bù rung lắc camera (GMC), tăng +36% BBox** |

> [!NOTE]
> **Nhận định khoa học cốt lõi:**
> 1. **Cơ chế Bù Rung Lắc Camera (GMC):** Khi drone chao liệng, BoT-SORT ước lượng ma trận chuyển động Affine để bù trừ vị trí dự đoán của Kalman Filter $\implies$ giảm thiểu hiện tượng văng BBox, hạ số lượng ID duy nhất từ **99–102 xuống 66–70 IDs** (giữ nguyên danh tính người đi bộ liên tục hơn).
> 2. **Sự Đánh Đổi Tính Toán (Trade-off):** Bước trích xuất điểm ảnh GMC của BoT-SORT ngốn thêm $25-40\text{ ms}$ trên CPU, kéo Live FPS xuống mức $\sim 18-27\text{ FPS}$ (đạt chuẩn realtime), trong khi ByteTrack so khớp ma trận hình học thuần túy nên đạt tốc độ vượt trội $\sim 52-84\text{ FPS}$.

---

### 2.3. Hiệu Năng Đột Phá Của Tầng Appearance Memory Bank & OCM

Để khắc phục nhược điểm nhảy ID khi đối tượng bị che khuất tạm thời hoặc drone chao liệng, hệ thống tích hợp tầng **Adaptive Appearance Memory Bank** (kết hợp phân rã 3 dải dọc khi $h/w \ge 1.65$ và 3 vành đồng tâm khi $h/w < 1.65$) cùng bộ dự phóng quán tính vận tốc thực tế **Observation-Centric Momentum (OCM)**.

---

### 2.4. Thực Nghiệm Đối Chứng Trên 5 Video Flycam Góc Nghiêng Oblique (27° – 59° — Chuẩn 50–60 FPS)

Thực nghiệm đo lường đối chứng giữa **ByteTrack Gốc** và **ByteTrack + Adaptive Memory Bank + OCM** sử dụng mô hình **`yolo26n_merged`** ở chế độ **`no-trail`** trên 5 video flycam chất lượng cao (3.715 frames, 126.258 detections):

| Video ID | Góc Nghiêng Thực Tế | Số Frame | Tổng Detections | Baseline ByteTrack (IDs) | Đề Xuất (+ Memory & OCM) | Tỷ Lệ Giảm Nhảy ID | Lượt Re-ID Khôi Phục | Live Tracking FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`video_01`** *(1345327468)* | $46.9^\circ$ | 511 | 46.877 | 656 IDs | **397 IDs** | **-39.48%** | 1.564 lượt | 30.2 FPS |
| **`video_02`** *(1975140282)* | $27.1^\circ$ | 955 | 36.448 | 440 IDs | **286 IDs** | **-35.00%** | 660 lượt | 49.1 FPS |
| **`video_03`** *(2272676305)* | $29.7^\circ$ | 1.439 | 19.926 | 414 IDs | **265 IDs** | **-35.99%** | 806 lượt | 45.2 FPS |
| **`video_04`** *(2281108543)* | $50.1^\circ$ | 541 | 22.507 | 625 IDs | **347 IDs** | **-44.48%** | 1.125 lượt | 40.3 FPS |
| **`video_05`** *(2288458623)* | $58.9^\circ$ | 269 | 500 | 48 IDs | **27 IDs** | **-43.75%** | 30 lượt | 61.6 FPS |
| **TỔNG CỘNG** | **$27.1^\circ - 58.9^\circ$** | **3.715** | **126.258** | **2.183 IDs** | **1.322 IDs** | **-39.44%** | **4.185 lượt** | **45.3 FPS** |

> [!NOTE]
> **Nhận định khoa học cốt lõi:**
> 1. **Hiệu quả cắt giảm phân mảnh vết (-39.44%):** Với đặc tính Recall $\sim 60-63\%$ của bài toán UAV cự ly trung bình, xác suất đối tượng xuất hiện liên tục 10 frames là $P \approx 0.63^{10} \approx 0.98\%$. Tầng OCM đón đầu quán tính kết hợp Memory Bank giúp phục hồi **4.185 lượt** đứt gãy vết, hạ tổng số ID từ 2.183 xuống 1.322 IDs.
> 2. **Tối ưu hiển thị HUD & Định danh:** Bảng Telemetry HUD siêu gọn $220 \times 84\text{ px}$ (giảm 60% diện tích), định danh ID tuần tự từ `#1` và nhãn BBox cỡ chữ $0.36$ giúp phi công dễ dàng quan sát mục tiêu mà không bị rối mắt.
> 3. **Tốc độ thời gian thực vượt trội:** Đạt trung bình **45.3 FPS** trên GPU cá nhân (độ trễ suy luận YOLO26n chỉ **9.26 – 13.31 ms**).

---

### 2.5. Giao Thức Khóa Mục Tiêu (Target Lock) & Điều Phối Bán Tự Hành (Semi-Autonomous FSM)

Kiến trúc vận hành (`src/target_protocol.py`) định hình chu trình điều khiển thông minh giữa Ground Station và Drone (Pi 5):
1. **Pha Tự Do (SEARCH_ALL_IDS):** Toàn bộ ID người trong khung hình đều được lưu trữ vector ngoại hình trong Memory Bank.
2. **Pha Khóa Mục Tiêu (TARGET_LOCKED):** Phi công chọn 1 ID $\to$ GCS gửi gói tin `LockTargetCommand` $\to$ Pi 5 xác nhận bằng gói `LockTargetAck` $\to$ Memory Bank chuyển sang chế độ Dedicated Memory cho mục tiêu này. Drone kích hoạt chế độ bay bán tự hành (hỗ trợ bám tâm nhưng vẫn ưu tiên quyền điều khiển từ tay cầm phi công).
3. **Pha Mất Dấu & Tìm Lại (TARGET_LOST_SPIN_SEARCH):** Nếu phi công lái đổi góc nhìn làm mất mục tiêu $\to$ Drone tự động thực hiện thao tác xoay $360^\circ$ quét tìm kiếm dựa trên template ngoại hình đã nhớ.
   - *Nếu tìm lại được:* Lập tức tái nhận diện (Re-ID) và tiếp tục bám vết (`TARGET_LOCKED`).
   - *Nếu hết thời gian xoay (Timeout):* Dừng xoay và chuyển sang trạng thái `STANDBY_HOVER` (giữ vị trí an toàn).

---

## 🛠️ 3. CÀI ĐẶT MÔI TRƯỜNG

Yêu cầu: **Python 3.10 – 3.12** và card đồ họa NVIDIA (khuyên dùng CUDA 12+).

```powershell
# 1. Clone repository và mở thư mục dự án:
git clone https://github.com/TheUn4given123jkbkjs/human-tracking-drone.git
cd human-tracking-drone

# 2. Khởi tạo và kích hoạt môi trường ảo:
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1

# 3. Cài đặt PyTorch CUDA 12.x và các thư viện phụ thuộc:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "ultralytics>=8.3.0" opencv-python numpy tqdm pandas
```

---

## 💻 4. HƯỚNG DẪN SỬ DỤNG (CLI REFERENCE)

Hệ thống hỗ trợ cơ chế **Dynamic Model Discovery** (tự động nhận diện toàn bộ mô hình có sẵn trong thư mục `models/`):

```powershell
# Kiểm tra danh sách mô hình hợp lệ có sẵn trên máy:
python track_drone.py --list-models
```

### 4.1. Cú Pháp Chạy Tổng Quát (General Syntax)

```powershell
python track_drone.py --model <tên_mô_hình|đường_dẫn_.pt> --tracker <bytetrack|botsort> --source <đường_dẫn_video> [tùy_chọn]
```

### 4.2. Bảng Tham Số Dòng Lệnh

| Tham Số | Kiểu | Mặc Định | Mô Tả & Tùy Chọn Hợp Lệ |
| :--- | :---: | :---: | :--- |
| **`--model`** | Chuỗi | `yolo26s_merged` | Tên model trong `models/` (`yolo26s_merged`, `yolo26n_merged`, `yolo26n_visdrone`) hoặc đường dẫn file `.pt`. |
| **`--tracker`** | Chuỗi | `bytetrack` | Thuật toán tracking: `bytetrack` (siêu tốc) hoặc `botsort` (bù rung lắc camera). |
| **`--source`** | Chuỗi | `data/test_drone_10_20m.mp4` | Đường dẫn file video `.mp4` đầu vào. |
| **`--conf`** | Float | `0.20` | Ngưỡng tin cậy phát hiện (cự ly xa: `0.15 - 0.20`; cự ly gần: `0.25 - 0.35`). |
| **`--iou`** | Float | `0.50` | Ngưỡng IoU ghép cặp liên kết vết. |
| **`--trail`** / **`--no-trail`** | Flag | `--no-trail` *(Mặc định tắt)* | Hiện hoặc ẩn vệt quỹ đạo di chuyển theo màu ID (mặc định tắt để khung hình quan sát của phi công thông thoáng). |
| **`--memory`** / **`--no-memory`** | Flag | `--memory` *(Bật)* | Bật hoặc tắt tầng Appearance Memory Bank (Re-ID đa vùng). |
| **`--sim-thresh`** | Float | `0.60` | Ngưỡng tương đồng Cosine để tái nhận diện (Re-ID) mục tiêu trong bộ nhớ. |
| **`--max-lost-frames`** | Int | `60` | Số frame tối đa lưu trữ mục tiêu bị mất dấu trước khi giải phóng khỏi RAM. |
| **`--step`** | Int | `1` | Bước nhảy frame: `1` (Full 30 FPS) hoặc `2` (Half-rate 15 FPS, giảm 50% tải GPU). |
| **`--max-frames`** | Int | `None` | Giới hạn số frame xử lý (ví dụ: `--max-frames 200` để test nhanh). |
| **`--display`** | Flag | `False` | Mở cửa sổ OpenCV GUI hiển thị quá trình bám vết trực tiếp. |
| **`--device`** | Chuỗi | `0` *(GPU)* | Thiết bị tính toán: `0` (GPU NVIDIA) hoặc `cpu`. |
| **`--out`** | Chuỗi | Tự động | Đường dẫn file hoặc thư mục lưu video (mặc định: `runs/track/<model>_<tracker>_<video>.mp4`). |

### 4.3. Ví Dụ Chạy Nhanh

#### Ví dụ 1: Chạy mô hình mặc định với ByteTrack xem trực tiếp trên màn hình
```powershell
python track_drone.py --model yolo26s_merged --tracker bytetrack --source data/test_drone_10_20m.mp4 --display
```

#### Ví dụ 2: Chạy BoT-SORT bù rung lắc camera (tự động xuất video vào `runs/track/`)
```powershell
python track_drone.py --model yolo26s_merged --tracker botsort --source data/test_drone_10_20m.mp4
```

---

## 📈 5. GIAO DIỆN BẢNG THÔNG SỐ HUD (HEADS-UP DISPLAY) & THỊ GIÁC

Bảng đo lường Telemetry HUD hiển thị trực tiếp ở góc trên bên trái video với thiết kế thu nhỏ $220 \times 84\text{ px}$ (giảm hơn 60% diện tích, độ trong suốt 70%, chữ sắc nét, không che khuất mục tiêu):
* **MODEL | TRACKER:** Tên mô hình AI và thuật toán bám vết đang thực thi.
* **FPS | Inf:** Tốc độ khung hình tức thời và độ trễ suy luận mạng nơ-ron (ms).
* **Active | Total IDs:** Số đối tượng đang bám vết đồng thời và tổng số ID duy nhất (được đánh số tuần tự từ `#1` cho từng video).
* **Re-ID:** Số lượt khôi phục danh tính thành công qua Memory Bank / Trạng thái TARGET LOCK.
* **BBox Labels:** Cỡ chữ gọn gàng (`font_scale=0.36`), badge nền ôm sát vừa vặn viền hộp.

---

## 📜 6. TRÍCH DẪN KHOA HỌC

* **Ultralytics Framework (v8.4.x):** One-to-One Head Native NMS-Free Real-time Detection Architecture.
* **ByteTrack:** *Zhang et al.*, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
* **BoT-SORT:** *Aharon et al.*, "BoT-SORT: Robust Associations Multi-Pedestrian Tracking", arXiv:2206.14651, 2022.
* **P-DESTRE:** *Kumar et al.*, "P-DESTRE: Pedestrian Detection, Tracking, and Re-Identification from Aerial Devices", IEEE TPAMI, 2021.
* **VisDrone-DET:** *Zhu et al.*, "Vision Meets Drones: A Challenge", IEEE TPAMI, 2021.
