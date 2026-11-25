# 🛸 UAV Human Tracking System — YOLO26 & ByteTrack / BoT-SORT

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26%20(NMS--Free)-00599C.svg?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![Trackers](https://img.shields.io/badge/Trackers-ByteTrack%20%7C%20BoT--SORT-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20OOP-orange.svg)]()

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

### 2.3. Hiệu Năng Đột Phá Của Tầng Appearance Memory Bank (Tái Nhận Diện Re-ID Đa Vùng)

Để khắc phục nhược điểm nhảy ID khi đối tượng bị che khuất tạm thời hoặc drone chao liệng, hệ thống được trang bị tầng **Appearance Memory Bank** (trích xuất vector màu HSV đa vùng không gian 3 tầng cơ thể: Đầu/Thân/Chân + so khớp Cosine Similarity lịch sử).

#### Bảng Đo Lường Hiệu Năng Khi Kích Hoạt Appearance Memory Bank (`test_drone_square_pdestre.mp4`):

| Cấu Hình Hệ Thống | Trạng Thái Memory Bank | Số Unique IDs | Tỷ Lệ Giảm ID Switches | Lượt Re-ID Khôi Phục | Live Tracking FPS | Nhận Xét Khoa Học |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`yolo26s_merged` + ByteTrack** | **TẮT** *(Gốc)* | 102 IDs | Baseline | 0 | ~52 – 80 FPS | Tốc độ cao nhưng đứt gãy vết liên tục |
| **`yolo26s_merged` + ByteTrack** | **BẬT (Memory Bank)** | **34 IDs** | **-66.7% (Đột phá)** | **134 lượt** | **~58 – 73 FPS** | **Sweet Spot: Giảm 2/3 số ID nhảy, tốc độ cực nhanh gấp 3.5 lần BoT-SORT** |
| **`yolo26s_merged` + BoT-SORT** | **TẮT** *(Gốc)* | 70 IDs | Baseline | 0 | ~18 – 27 FPS | Bù rung lắc tốt nhưng FPS thấp |
| **`yolo26s_merged` + BoT-SORT** | **BẬT (Memory Bank)** | **33 IDs** | **-52.9%** | **83 lượt** | ~17 – 20 FPS | Độ ổn định danh tính cao nhất (33 IDs), phù hợp máy trạm GPU mạnh |

#### Đánh Giá Tổng Thể Trên Toàn Bộ Kho 20 Video Thực Nghiệm (15 Video Val P-DESTRE + 5 Video Benchmark):
- **Quy mô mẫu kiểm định:** **20 video** chuẩn hóa (1.214 frames, 8.961 bounding boxes người).
- **Tổng số Unique IDs:** Giảm từ **1.399 IDs** xuống còn **712 IDs** (giảm **-49.1%** hiện tượng phân mảnh quỹ đạo trên toàn bộ 20 video).
- **Tổng số lượt Re-ID khôi phục:** **1.647 lượt** cứu vết thành công nhờ bộ nhớ đặc trưng HSV 3 vùng không gian cơ thể.
- **Tốc độ trung bình toàn hệ thống:** Duy trì **~16 – 59 FPS** (Live trung bình ~22.5 FPS), độ trễ suy luận YOLO26s chỉ **8.9 – 20.8 ms**, đảm bảo thời gian thực tuyệt đối.
- Báo cáo số liệu chi tiết từng video: `runs/benchmark_20_videos_report.csv`.

---

### 2.4. Giao Thức Khóa Mục Tiêu (Target Lock) & Điều Phối Bán Tự Hành (Semi-Autonomous FSM)

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
| **`--trail`** / **`--no-trail`** | Flag | `--trail` *(Bật)* | Hiện hoặc ẩn vệt quỹ đạo di chuyển theo màu ID. |
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

## 📈 5. GIAO DIỆN BẢNG THÔNG SỐ HUD (HEADS-UP DISPLAY)

Bảng đo lường Telemetry HUD hiển thị trực tiếp ở góc trên bên trái video:
* **MODEL / TRACKER:** Tên mô hình AI và thuật toán bám vết đang thực thi.
* **LIVE FPS / LATENCY:** Tốc độ khung hình tức thời và độ trễ suy luận mạng nơ-ron (ms).
* **NMS-FREE:** Độ trễ hậu xử lý Native NMS-Free ($\le 1.0\text{ ms}$).
* **ACTIVE / UNIQUE IDS:** Số người đang bám vết đồng thời và tổng số ID duy nhất trong video.

---

## 📜 6. TRÍCH DẪN KHOA HỌC

* **Ultralytics Framework (v8.4.x):** One-to-One Head Native NMS-Free Real-time Detection Architecture.
* **ByteTrack:** *Zhang et al.*, "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022.
* **BoT-SORT:** *Aharon et al.*, "BoT-SORT: Robust Associations Multi-Pedestrian Tracking", arXiv:2206.14651, 2022.
* **P-DESTRE:** *Kumar et al.*, "P-DESTRE: Pedestrian Detection, Tracking, and Re-Identification from Aerial Devices", IEEE TPAMI, 2021.
* **VisDrone-DET:** *Zhu et al.*, "Vision Meets Drones: A Challenge", IEEE TPAMI, 2021.
