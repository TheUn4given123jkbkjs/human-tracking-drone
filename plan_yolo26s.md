# 📋 Kế Hoạch Huấn Luyện YOLO26s (GPU Trạm Mặt Đất) — Run 3

> **Đề tài:** Nhận diện và theo dõi người bằng máy bay không người lái (UAV Human Detection & Tracking)  
> **Mục tiêu mô hình:** YOLO26s (Small — 9.5M params, Native NMS-Free)  
> **Kế thừa dữ liệu:** 100% dữ liệu đã xử lý sạch từ Giai đoạn 2 (VisDrone + P-DESTRE)  
> **Tệp thực thi Colab mới:** `train_yolo26s_on_colab.ipynb` *(Độc lập, không ghi đè notebook YOLO26n cũ)*

---

## 1. MỤC TIÊU KHOA HỌC & TIÊU CHÍ NGHIỆM THU (DoD)

1. **Hiệu năng nhận diện:**
   - Đạt $mAP_{50} \ge \mathbf{66.5\%}$ trên tập đánh giá hợp nhất (kỳ vọng tăng $+8.2\%$ so với YOLO26n Nano).
   - Đạt $mAP_{50-95} \ge \mathbf{35.0\%}$.
   - Khả năng bắt người ở xa ($15 - 20\text{m}$, kích thước $< 32\text{px}$) và người bị che khuất một phần tăng tối thiểu $+10\%$ Recall.
2. **Hiệu năng thời gian thực:**
   - Độ trễ suy luận (Inference Latency): $\le \mathbf{3.5\text{ ms/frame}}$ trên GPU trạm mặt đất (RTX 3060/4060).
   - Độ trễ hậu xử lý: $\le \mathbf{0.35\text{ ms}}$ (Native NMS-Free).
3. **Bảo toàn dữ liệu & Chống rò rỉ:**
   - 100% độc lập giữa tập Train và Val theo cấp độ chuỗi video (*Sequence-level split*).
   - Không ghi đè hay làm xáo trộn bất kỳ file nào của Giai đoạn 1 & 2.

---

## 2. NGUYÊN TẮC THIẾT KẾ DỮ LIỆU & TỐI ƯU HÓA COLAB

- **Tận dụng 100% tài nguyên đã bóc tách trên Google Drive:**
  - Không cần tải lại hay giải nén file thô `dataset.tar` (38.68 GB).
  - Tận dụng trực tiếp:
    - `data/visdrone_filtered_yolo.zip` (1.46 GB — 3.739 ảnh sạch).
    - `data/pdestre_processed/*.zip` (75 sequence zips đã chia sẵn 60 train / 15 val).
  - Bung trực tiếp vào SSD Colab (`/content/datasets/`) chỉ mất $\sim 2$ phút.
- **Tập dữ liệu huấn luyện hợp nhất (Unified Dataset):**
  - **Train:** 3.299 ảnh VisDrone sạch + toàn bộ ảnh từ 60 video P-DESTRE train (lấy mẫu 2 FPS).
  - **Val đối chứng đa miền:** 440 ảnh VisDrone Val (đối chứng không quên tri thức) + 15 video P-DESTRE Val (đối chứng góc dốc $45^\circ - 90^\circ$).

---

## 3. CẤU HÌNH SIÊU THAM SỐ HUẤN LUYỆN (HYPERPARAMETERS)

| Tham số | Giá trị thiết lập | Cơ sở khoa học |
| :--- | :---: | :--- |
| **Model khởi tạo** | `yolo26s.pt` | Kiến trúc Small 9.5M params, trích xuất đặc trưng sâu hơn Nano 4 lần |
| **Độ phân giải (`imgsz`)** | `640` | Đối chứng chuẩn mực với Run 1 và Run 2 (có cell test riêng `960`) |
| **Số Epochs** | `80` | Đảm bảo hội tụ sâu cho mạng 9.5M params |
| **Batch Size** | `16` hoặc `32` | Tối ưu dung lượng VRAM 16GB của Colab T4/A100 |
| **Learning Rate** | `lr0=0.01`, `lrf=0.01` | Kết hợp `cos_lr=True` để hạ nhiệt độ học mượt mà |
| **Mosaic Augmentation** | Bật, tắt ở 10 epoch cuối | `close_mosaic=10` giúp tinh chỉnh bounding box chuẩn xác |
| **Thư mục lưu weights** | Google Drive | `human_tracking_drone/models/yolo26s_merged_run3/weights/best.pt` |

---

## 4. PHÂN RÃ GÓI CÔNG VIỆC (WORK PACKAGES)

- **WP1: Xây dựng Notebook `train_yolo26s_on_colab.ipynb` độc lập:**
  - Khởi tạo notebook mới, cấu hình tự động giải nén từ Drive SSD, load `yolo26s.pt`, huấn luyện 80 epochs và tự động lưu `best.pt`.
- **WP2: Thực thi Huấn luyện trên Colab:**
  - Chạy toàn bộ tiến trình trên Google Colab GPU, theo dõi đường cong loss và mAP qua TensorBoard/CSV.
- **WP3: Đánh giá Đối chứng Đa miền:**
  - Đánh giá độc lập trên VisDrone Val và P-DESTRE Val, trích xuất ma trận nhầm lẫn (`confusion_matrix.png`).
- **WP4: Tích hợp vào CLI Local:**
  - Kéo `best.pt` về máy local (`models/yolo26s_merged_run3/weights/best.pt`), bổ sung alias `run3` vào `src/config.py` và `track_drone.py`.
