# 🛸 UAV Human Tracking System — YOLO26n & ByteTrack / BoT-SORT

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26n-00599C.svg?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![Trackers](https://img.shields.io/badge/Trackers-ByteTrack%20%7C%20BoT--SORT-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20OOP-orange.svg)]()

> **Đề tài NCKH:** Nghiên cứu và Triển khai Thuật toán Nhận diện & Bám vết Người Thời gian thực cho Thiết bị Bay Không người lái (UAV) sử dụng Kiến trúc YOLO26n Native NMS-Free kết hợp Thuật toán Bám vết Đa đối tượng (ByteTrack & BoT-SORT).

---

## 📌 1. ĐIỂM NỔI BẬT KHOA HỌC & CÔNG NGHỆ

1. **Kiến trúc Native NMS-Free (YOLO26n Nano — 2.5M tham số):**
   - Loại bỏ hoàn toàn tầng lọc phi cực đại *Non-Maximum Suppression (NMS)* truyền thống vốn là "nút thắt cổ chai" tính toán trên ảnh flycam có mật độ người cao.
   - Hậu xử lý siêu tốc: **$0.28\text{ ms/frame}$**, tốc độ toàn chuỗi đạt **~95 FPS** trên GPU máy tính, duy trì ổn định $\ge 55\text{ FPS}$ ngay cả ở các khung cảnh đông đúc $>45$ người/khung hình.
2. **Chuẩn hóa Thực nghiệm Đa tầng Góc nhìn & Độ cao:**
   - Lọc sạch 81.540 vùng nhiễu *ignored* và đối tượng người ngoài tầm hiệu dụng ($>20\text{ m}$) trên tập VisDrone-DET.
   - Hợp nhất dữ liệu chuỗi video 4K góc dốc $45^\circ - 90^\circ$ từ tập dữ liệu P-DESTRE, giải quyết triệt để điểm mù nhận diện đỉnh đầu/vai của camera drone.
3. **Chiến lược Triển khai Kép (Hybrid Deployment):**
   - **Ground Station (Máy trạm mặt đất):** Triển khai dòng **YOLO26s** (9.5M tham số) cho độ chính xác cao nhất ($mAP_{50} \ge 66.5\%$).
   - **Onboard Edge (Máy tính biên trên Drone):** Triển khai dòng siêu nhẹ **YOLO26n** (2.5M tham số, TensorRT FP16/INT8) hoạt động độc lập làm cơ chế an toàn bay (*Fail-safe*) tự động duy trì quỹ đạo bám người khi mất sóng viễn thông.

---

## 📊 2. KẾT QUẢ THỰC NGHIỆM ĐỐI CHỨNG

Kết quả đo lường độc lập theo phương pháp luận thực nghiệm khoa học:

| Giai đoạn Thực nghiệm | Mô hình Trọng Số | Tập Dữ Liệu Kiểm Thử | $mAP_{50}$ | $mAP_{50-95}$ | Precision | Recall | Độ Trễ Post-process | Trạng Thái |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Giai đoạn 1 (Baseline)** | YOLO26n | VisDrone Val (440 ảnh) | **57.26%** | **26.91%** | **58.40%** | **56.38%** | **0.28 ms** | ✅ Hoàn tất |
| **Giai đoạn 2 (Hợp nhất Đợt 1)** | YOLO26n | **VisDrone Val** *(Đối chứng)* | **58.30%** *(+1.04%)* | **27.77%** *(+0.86%)* | **59.28%** | **59.09%** *(+2.71%)* | **0.28 ms** | ✅ Không quên kiến thức cũ |
| **Giai đoạn 2 (Hợp nhất Đợt 1)** | YOLO26n | **P-DESTRE Val** *(Góc dốc $45^\circ - 90^\circ$)* | **88.36%** | **54.34%** | **82.40%** | **86.52%** | **0.28 ms** | 🎯 **Đột phá vượt mục tiêu 65%** |
| **Giai đoạn 2+ (Bổ sung Đợt 2–4)** | YOLO26n | Full 75 videos P-DESTRE / imgsz 960 | *Aim: $\ge 60.0\%$* | *Aim: $\ge 29.0\%$* | — | — | **< 0.30 ms** | ⏳ Sẵn sàng kích hoạt |
| **Giai đoạn 3 (Ground Station)** | YOLO26s | VisDrone + P-DESTRE | *Aim: $\ge 66.5\%$* | *Aim: $\ge 35.0\%$* | — | — | **~0.35 ms** | 📋 Lộ trình tiếp theo |

> [!NOTE]
> Xem báo cáo phân tích chi tiết, ma trận đối sách thực tế và lộ trình NCKH tại: [docs/RESEARCH_ROADMAP.md](file:///d:/record_by_me/NCKH/human_tracking_drone/docs/RESEARCH_ROADMAP.md).

---

## 🏛️ 3. KIẾN TRÚC MÃ NGUỒN HƯỚNG ĐỐI TƯỢNG (OOP ARCHITECTURE)

Hệ thống được thiết kế theo nguyên tắc *Single Responsibility Principle*, phân rã thành các module độc lập trong thư mục [src/](file:///d:/record_by_me/NCKH/human_tracking_drone/src):

```text
human_tracking_drone/
├── src/                                  # Package hướng đối tượng độc lập (OOP)
│   ├── __init__.py                       # Export các interface chính của hệ thống
│   ├── config.py                         # Cấu hình registry mô hình ('run1', 'run2') & tracker ('bytetrack', 'botsort')
│   ├── metrics.py                        # Class TrackingMetrics: Đo lường khoa học FPS, độ trễ, số ID, active targets
│   ├── detector.py                       # Class UAVDetector: Đóng gói YOLO26n Native NMS-Free (load weights, suy luận)
│   ├── tracker.py                        # Class UAVTracker: Đóng gói thuật toán bám vết (ByteTrack & BoT-SORT)
│   ├── visualizer.py                     # Class TrackingVisualizer: Bật/tắt vệt quỹ đạo di chuyển, vẽ HUD Telemetry
│   └── pipeline.py                       # Class UAVTrackingPipeline: Điều phối toàn bộ luồng video I/O và số liệu
├── track_drone.py                        # CLI Entrypoint linh hoạt điều khiển qua PowerShell
├── train_yolo26n_on_colab.ipynb          # Notebook điều phối huấn luyện tự động trên Google Colab
├── configs/                              # File cấu hình dataset YAML cho Ultralytics
├── data/                                 # Hệ thống dữ liệu & video kiểm thử (Xem data/README.md)
│   ├── README.md                         # Đặc tả siêu dữ liệu chi tiết (Metadata Raw/Processed & Drive Links)
│   ├── test_drone_campus_15m.mp4         # Drone khuôn viên trường (10-15m, góc dốc 40°, 45+ người/frame)
│   ├── test_drone_square_pdestre.mp4     # Drone quảng trường ngoài trời (P-DESTRE, 10-15m, góc 45°)
│   ├── test_drone_street_pdestre.mp4     # Drone đường phố rợp bóng cây (P-DESTRE, 10-15m, góc 40°)
│   ├── test_drone_walkway_pdestre.mp4    # Drone lối đi bộ cự ly gần (P-DESTRE, 10-12m, góc 45°)
│   ├── demo_drone_visdrone.mp4           # Drone phố đô thị đông đúc (VisDrone, 10-15m, góc 35°)
│   ├── test_drone_10_20m.mp4             # Drone góc nhìn 10-20m bao quát
│   └── test_drone_10_20m_alt.mp4         # Drone góc nghiêng 45° quét toàn cảnh
├── models/                               # Checkpoints weights tốt nhất
│   ├── yolo26n_visdrone_filtered_run1/   # Weights Run 1 Baseline (57.26%)
│   └── yolo26n_merged_run2/              # Weights Run 2 Merged (58.30% VisDrone / 88.36% P-DESTRE)
├── runs/track/                           # Video xuất kết quả sau khi theo dõi (.mp4) — Xem lưu trữ Drive bên dưới
├── docs/                                 # Tài liệu kỹ thuật chuyên sâu & lộ trình nghiên cứu
└── .gitignore                            # Loại trừ file rác, dữ liệu nặng và cache
```

---

## 🛠️ 4. CÀI ĐẶT MÔI TRƯỜNG

Yêu cầu: **Python 3.10+** và card đồ họa NVIDIA (khuyên dùng CUDA 12+):

```bash
# 1. Clone repository hoặc mở thư mục dự án:
cd d:\record_by_me\NCKH\human_tracking_drone

# 2. Cài đặt các thư viện phụ thuộc:
pip install "ultralytics>=8.3.0" opencv-python numpy tqdm pandas
```

> [!NOTE]
> **Về thư mục `runs/`:** Các video demo kết quả tracking đã thực thi được lưu trữ đối chứng trên [Google Drive: Thư mục runs/track](https://drive.google.com/drive/folders/1SMPglsrYyraLgBkybQVk-oqxFZixxS_T). Bạn **không cần tải về** thư mục `runs/` khi cài đặt và chạy thử nghiệm. Khi thực thi lệnh `track_drone.py`, chương trình sẽ tự động tạo thư mục `runs/track/` và xuất video trực tiếp tại máy local.

---

## 💻 5. HƯỚNG DẪN SỬ DỤNG QUA POWERSHELL (CLI REFERENCE)

### 5.1. Cú pháp Tổng quát
```powershell
python track_drone.py [Cấu hình Mô hình] [Thuật toán Track] [Video Nguồn] [Tùy chỉnh Hiển thị & Tối ưu]
```
*(Các tham số đều là Named Arguments bắt đầu bằng dấu `--`, có thể nhập theo thứ tự bất kỳ).*

### 5.2. Bảng Tra cứu Chi tiết Từng Tham số

| Tham Số | Tên / Ý Nghĩa | Kiểu | Mặc Định | Mô Tả & Tùy Chọn Hợp Lệ |
| :--- | :--- | :---: | :---: | :--- |
| **`--trail`**<br>hoặc **`--no-trail`** | **Hiện hoặc Ẩn vệt quỹ đạo di chuyển** | Flag | `--trail` *(Bật)* | • **`--trail`**: Bật vệt đường lịch sử di chuyển theo màu ID.<br>• **`--no-trail`**: Ẩn vệt đường, chỉ hiện BBox và ID (rất hữu ích khi cảnh quá đông người). |
| **`--model`** | Phiên bản trọng số mô hình | Chuỗi | `run2` | • **`run2`**: Trọng số tối ưu Giai đoạn 2 (`yolo26n_merged_run2`).<br>• **`run1`**: Trọng số Baseline Giai đoạn 1 (`yolo26n_visdrone_filtered_run1`).<br>• Hoặc đường dẫn trực tiếp tới file `.pt` bất kỳ. |
| **`--tracker`** | Thuật toán bám vết đa đối tượng | Chuỗi | `bytetrack` | • **`bytetrack`**: Thuật toán Kalman 2 tầng (Siêu tốc, khuyên dùng).<br>• **`botsort`**: Thuật toán tích hợp bù rung lắc camera drone (CMC). |
| **`--source`** | Đường dẫn video đầu vào | Chuỗi | `data/test_drone_10_20m.mp4` | Đường dẫn tới file `.mp4` flycam cần kiểm thử (xem mục 5.3). |
| **`--conf`** | Ngưỡng tin cậy phát hiện (Confidence) | Float | `0.20` | Từ `0.05` đến `0.90`. Cự ly xa nên để `0.15 - 0.20`; cự ly gần để `0.25 - 0.35`. |
| **`--iou`** | Ngưỡng IoU ghép cặp tracking | Float | `0.50` | Từ `0.3` đến `0.7`. |
| **`--step`** | Bước nhảy khung hình (Frame Step) | Int | `1` | • **`1`**: Full 30 FPS trên từng khung hình.<br>• **`2`**: Half-rate (15 FPS Detection + 30 FPS Kalman Predict, giảm 50% tải GPU). |
| **`--max-frames`** | Giới hạn số frame xử lý | Int | `None` *(Chạy hết)* | Ví dụ: `--max-frames 150` để kiểm thử nhanh 150 frame đầu. |
| **`--display`** | Xem video trực tiếp trên màn hình | Flag | `False` | Mở cửa sổ OpenCV GUI hiển thị quá trình bám vết trực tiếp. |
| **`--device`** | Thiết bị tính toán phần cứng | Chuỗi | `0` *(GPU)* | • **`0`**: Chạy bằng card đồ họa GPU NVIDIA.<br>• **`cpu`**: Chạy bằng CPU. |
| **`--out`** | Đường dẫn file video kết quả | Chuỗi | Tự động | Tự động lưu vào `runs/track/track_{model}_{tracker}_{video}.mp4`. |

---

### 5.3. Các Câu Lệnh Mẫu Thực Nghiệm Thông Dụng

#### Kịch bản 1: Mô hình Mới (Run 2) + ByteTrack + Tắt vệt đường tracking (Gọn gàng):
```powershell
python track_drone.py --model run2 --tracker bytetrack --source data/test_drone_campus_15m.mp4 --no-trail
```

#### Kịch bản 2: Mô hình Mới (Run 2) + ByteTrack + Bật vệt đường tracking (Mặc định):
```powershell
python track_drone.py --model run2 --tracker bytetrack --source data/test_drone_campus_15m.mp4 --trail
```

#### Kịch bản 3: Mô hình Mới (Run 2) + BoT-SORT (Bù trừ rung lắc camera drone):
```powershell
python track_drone.py --model run2 --tracker botsort --source data/test_drone_campus_15m.mp4 --no-trail
```

#### Kịch bản 4: So sánh đối chứng Mô hình Cũ (Baseline Run 1) + ByteTrack:
```powershell
python track_drone.py --model run1 --tracker bytetrack --source data/test_drone_campus_15m.mp4 --no-trail
```

#### Kịch bản 5: Chạy thử nghiệm trên video quảng trường ngoài trời góc $45^\circ$:
```powershell
python track_drone.py --model run2 --tracker bytetrack --source data/test_drone_square_pdestre.mp4 --conf 0.20 --trail
```

#### Kịch bản 6: Chế độ tiết kiệm pin trên Drone (Half-rate 15 FPS) + Giới hạn 200 frames:
```powershell
python track_drone.py --model run2 --tracker bytetrack --source data/test_drone_campus_15m.mp4 --step 2 --max-frames 200 --no-trail
```

---

## 📈 6. GIAO DIỆN BẢNG THÔNG SỐ HUD (HEADS-UP DISPLAY)

Khi xuất video hoặc bật cờ `--display`, hệ thống tự động chèn bảng HUD Telemetry bán trong suốt ở góc trên bên trái:
* **MODEL:** Tên mô hình đang chạy (`yolo26n_merged_run2` hoặc `yolo26n_visdrone_run1`).
* **TRACKER:** Tên thuật toán bám vết (`BYTETRACK` hoặc `BOTSORT`).
* **LIVE FPS:** Tốc độ khung hình xử lý thời gian thực tức thời.
* **LATENCY:** Độ trễ suy luận của mạng nơ-ron (Inference ms).
* **NMS-FREE:** Độ trễ hậu xử lý Native NMS-Free (thường chỉ $0.25 - 0.35\text{ ms}$).
* **ACTIVE:** Số lượng người đang được theo dõi đồng thời trong khung hình hiện tại.
* **TOTAL UNIQUE IDS:** Tổng số định danh người duy nhất đã được bám vết trong suốt video.

---

## 📜 7. TRÍCH DẪN & TÀI LIỆU KHOA HỌC

Đề tài được phát triển và đối chứng thực nghiệm theo tiêu chuẩn nghiên cứu khoa học:
* Ultralytics YOLO26: State-of-the-art Real-time Object Detection with Native NMS-Free Architecture.
* ByteTrack: Multi-Object Tracking by Associating Every Detection Box (ECCV 2022).
* BoT-SORT: Robust Associations Multi-Pedestrian Tracking (arXiv 2022).
* P-DESTRE: A Fully Annotated Dataset for Pedestrian Detection, Tracking, and Re-Identification from Aerial Devices (IEEE T-PAMI).
* VisDrone-DET: A Benchmark and Dataset for Object Detection in Drone-captured Images.
