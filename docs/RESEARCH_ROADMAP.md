# 📑 Lộ Trình Nghiên Cứu & Ma Trận Đối Sách — UAV Human Tracking

> **Đề tài:** Nhận diện và theo dõi người bằng máy bay không người lái (UAV Human Detection & Tracking)  
> **Kiến trúc:** YOLO26n Native NMS-Free kết hợp ByteTrack & BoT-SORT  
> **Cập nhật:** 23/11/2026

---

## 1. MỤC TIÊU & KIẾN TRÚC KỸ THUẬT

### 1.1. Thách Thức Kỹ Thuật
- **Đối tượng siêu nhỏ:** Người nhìn từ độ cao 10–20m thường có kích thước $< 32 \times 32\text{ px}$.
- **Biến thiên góc quay:** Góc quan sát thay đổi liên tục từ góc xiên ($15^\circ - 45^\circ$) đến dốc thẳng đứng ($60^\circ - 90^\circ$ — Bird's-Eye View), làm mất hoàn toàn dáng người và chân.
- **Ràng buộc thời gian thực:** Camera flycam chuẩn $30\text{ FPS}$ yêu cầu toàn bộ chuỗi (Inference + Post-process + Association) phải xử lý trong $\le 33.3\text{ ms/frame}$.

### 1.2. Giải Pháp Công Nghệ
- **YOLO26n (Nano — 2.5M params):**
  - Cơ chế **Native NMS-Free** loại bỏ hoàn toàn tầng Non-Maximum Suppression truyền thống.
  - Hậu xử lý phẳng ở mức **$0.28\text{ ms/frame}$**, giải quyết hiện tượng nghẽn cổ chai khi mật độ người đông.
- **ByteTrack & BoT-SORT (Tracking-by-Detection):**
  - ByteTrack tận dụng cả detection điểm tin cậy thấp để bảo toàn vết khi người bị che khuất một phần.
  - BoT-SORT tích hợp bù trừ chuyển động máy ảnh (CMC) hỗ trợ ổn định khi drone lượn vòng.
- **Mã nguồn OOP Module hóa:** Toàn bộ luồng xử lý được đóng gói trong thư mục `src/`, điều khiển linh hoạt qua CLI `track_drone.py`.

---

## 2. KẾT QUẢ THỰC NGHIỆM ĐÃ ĐẠT ĐƯỢC

### 2.1. Bảng Số Liệu Đối Chứng Đa Chiều

| Giai đoạn Thực nghiệm | Mô hình | Tập Dữ Liệu Đánh Giá | $mAP_{50}$ | $mAP_{50-95}$ | Precision | Recall | Post-process | Trạng thái |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Giai đoạn 1 (Baseline)** | YOLO26n | VisDrone Val (440 ảnh) | **57.26%** | **26.91%** | **58.40%** | **56.38%** | **0.28 ms** | ✅ Hoàn tất |
| **Giai đoạn 2 (Hợp nhất)** | YOLO26n | **VisDrone Val** *(Đối chứng)* | **58.30%** *(+1.04%)* | **27.77%** *(+0.86%)* | **59.28%** | **59.09%** *(+2.71%)* | **0.28 ms** | ✅ Không quên tri thức cũ |
| **Giai đoạn 2 (Hợp nhất)** | YOLO26n | **P-DESTRE Val** *(Góc $45^\circ - 90^\circ$)* | **88.36%** | **54.34%** | **82.40%** | **86.52%** | **0.28 ms** | 🎯 Vượt xa mục tiêu 65% |

### 2.2. Điểm Nhấn Đột Phá
1. **Triệt tiêu hiện tượng quên kiến thức cũ (*No Catastrophic Forgetting*):** Khi nạp thêm video P-DESTRE góc dốc, mô hình không suy giảm mà còn tăng $mAP_{50}$ trên tập VisDrone Val ($57.26\% \to 58.30\%$).
2. **Khắc phục điểm mù góc dốc:** Độ nhạy nhận diện trên P-DESTRE đạt **$mAP_{50} = 88.36\%$** và Recall **$86.52\%$**, xử lý hiệu quả trường hợp người chỉ lộ đỉnh đầu và vai.
3. **Hiệu năng thời gian thực:** Trọng số `best.pt` (5.2 MB) đạt tốc độ toàn chuỗi **~95 FPS** trên GPU, sẵn sàng triển khai trên máy tính nhúng hoặc trạm mặt đất.

---

## 3. MA TRẬN ĐỐI SÁCH CHO CÁC TÌNH HUỐNG THỰC TẾ

| Tình huống Thách thức | Hiện tượng & Nguyên nhân | Đối sách Kỹ thuật | Phương án Dự phòng |
| :--- | :--- | :--- | :--- |
| **1. Che khuất tạm thời (Occlusion)** | Cây cối, cột đèn hoặc người đi cắt ngang làm mất dấu 0.5–2s. | Tăng `track_buffer` lên 60 frames (bảo lưu vết 2s); hạ ngưỡng vòng 2 `low_thresh=0.1`. | Tích hợp trích xuất đặc trưng ngoại hình (Re-ID vector) từ backbone YOLO26 để so khớp Cosine. |
| **2. Góc nhìn đỉnh đầu ($60^\circ - 90^\circ$)** | Cơ thể thu hẹp thành đỉnh đầu/vai; dễ nhầm với cọc tiêu, thùng rác. | Bổ sung dữ liệu P-DESTRE góc $90^\circ$; tăng cường xoay đa hướng (`degrees=15.0`). | Bộ lọc hậu xử lý: Loại bỏ bbox dạng tĩnh không có vector vận tốc từ Kalman Filter. |
| **3. Rung lắc Gimbal & Motion Blur** | Drone đổi hướng đột ngột làm ảnh nhòe mờ, bbox trôi dạt. | Áp dụng augment Motion Blur khi train; mở rộng hiệp phương sai sai số $Q$ trong Kalman. | Sử dụng BoT-SORT với tính năng bù trừ chuyển động camera (CMC). |
| **4. Rớt sóng viễn thông (Link Failure)** | Mất kết nối vô tuyến giữa Drone và trạm mặt đất. | **Kiến trúc Kép:** Trạm mặt đất chạy YOLO26s; máy tính biên drone chạy YOLO26n độc lập. | Kích hoạt Fail-safe: Drone tự duy trì bám mục tiêu bằng YOLO26n và tự động quay về (RTH). |
| **5. Cân bằng FPS & Tải tính toán** | Chạy full 30 FPS ngốn pin; giảm xuống 2-3 FPS làm trôi Kalman. | Chạy Full 30 FPS (do YOLO26n chỉ tốn 10ms); hoặc chạy Half-rate 15 FPS (xen kẽ YOLO và Kalman Predict). | Điều chỉnh tham số `step` trong CLI để cân đối giữa độ mượt quỹ đạo và công suất tiêu thụ. |

---

## 4. MA TRẬN SO SÁNH CÁC MÔ HÌNH DỰ ĐỊNH

### 4.1. So Sánh Dòng YOLO26 Theo Quy Mô
| Mô hình | Params | FLOPs (640p) | NMS Overhead | Mục tiêu $mAP_{50}$ | Latency (GPU T4) | Nền tảng Triển khai |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO26n (Nano)** | **2.5M** | **~7.5G** | **0 ms** | **$\ge 60.0\%$** | **~1.2 ms** | **Máy tính biên Drone (Jetson / RPi5)** |
| **YOLO26s (Small)** | 9.5M | ~26.2G | 0 ms | $\ge 66.5\%$ | ~2.8 ms | Trạm mặt đất (Ground Station) |
| **YOLO26m (Medium)**| 20.8M | ~68.0G | 0 ms | $\ge 71.0\%$ | ~6.5 ms | Server trạm điều khiển trung tâm |

### 4.2. Khảo Sát Triệt Tiêu: NMS-Free vs DFL + NMS
- **YOLOv8n / YOLOv11n (DFL + NMS):** Khi số người $> 50$ người/frame, độ trễ NMS tăng vọt từ $1.5\text{ ms}$ lên $> 10\text{ ms}$, gây nghẽn hệ thống.
- **YOLO26n (Native NMS-Free):** Trả về kết quả trực tiếp, độ trễ hậu xử lý phẳng ở mức **$0.28\text{ ms}$**, ổn định bất kể mật độ đám đông.

### 4.3. So Sánh Thuật Toán Bám Vết (Tracking Benchmark)
| Thuật toán | Cơ chế liên kết | Độ phức tạp | Tốc độ | Khả năng xử lý che khuất | Phù hợp Drone |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **ByteTrack** | Kalman Filter + 2-stage IoU | Thấp ($O(N)$) | **~95 - 120 FPS** | Rất tốt (nhờ low-score) | **Tối ưu thời gian thực** |
| **BoT-SORT** | Kalman Filter + Camera Motion Compensation | Trung bình | **~45 - 60 FPS** | Tốt (bù rung lắc camera) | **Tốt cho bay lượn nhanh** |
| DeepSORT | CNN Re-ID Feature Extractor | Cao ($O(N \times D)$) | ~18 - 25 FPS | Xuất sắc | Kém (không đạt 30 FPS) |

---

## 5. KẾ HOẠCH TRIỂN KHAI TIẾP THEO

1. **Thực nghiệm đo đạc đối chứng 4 kịch bản tracking trên PowerShell:**
   - Chạy chéo 2 mô hình (`run1`, `run2`) với 2 thuật toán (`bytetrack`, `botsort`) trên video khuôn viên `test_drone_campus_15m.mp4`.
   - Ghi nhận các chỉ số: FPS trung bình, độ trễ từng khâu, số lượng ID duy nhất và tỷ lệ ID Switch.
2. **Nâng cấp độ phân giải và dữ liệu mở rộng:**
   - Thử nghiệm đánh giá với độ phân giải `imgsz=960` để tối ưu cho người kích thước nhỏ.
   - Chuẩn bị bổ sung các đợt bóc tách tiếp theo từ P-DESTRE nếu cần đẩy $mAP_{50}$ VisDrone vượt $\ge 60\%$.
3. **Thử nghiệm mô hình YOLO26s:** Huấn luyện phiên bản Small cho Trạm mặt đất nhằm nâng cao độ chính xác nhận diện.
