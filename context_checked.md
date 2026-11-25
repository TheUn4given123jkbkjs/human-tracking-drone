# Context Checkpoint — Nghiên cứu Khoa học
> **Cập nhật ngày:** 25/11/2026 (Nâng cấp WP6: Cơ chế Appearance Memory Bank & Giao thức Target Lock/Re-acquire)

- **Tên Đề tài / Mục tiêu:** Nhận diện và theo dõi người bằng máy bay không người lái (UAV Human Detection & Tracking) sử dụng kiến trúc YOLO26 Native NMS-Free kết hợp ByteTrack & BoT-SORT cải tiến thêm lớp Appearance Memory Bank.
- **Vấn đề Khoa học & Giả thuyết ($H_0/H_1$):**
  - $H_0$: Các mô hình thị giác truyền thống bị nghẽn độ trễ do tầng NMS khi mật độ đông, mất dấu khi camera drone chao liệng, và các thuật toán tracking thuần hình học (IoU/Kalman) bị đứt gãy, nhảy ID liên tục khi đối tượng bị che khuất tạm thời hoặc đổi góc nhìn đột ngột.
  - $H_1$: Kiến trúc YOLO26 Native NMS-Free kết hợp dữ liệu hợp nhất (VisDrone + P-DESTRE) duy trì độ trễ phẳng ($< 1.0\text{ ms}$), kết hợp lớp **Appearance Memory Bank** (lưu trữ vector đặc trưng màu/không gian đa vùng và cosine matching) giúp tái nhận diện (Re-ID) và khôi phục ID bị mất dấu, giảm thiểu tối đa hiện tượng phân mảnh quỹ đạo (ID switches).

---

### Các Gói Công việc (Work Packages):
- **WP1 (Hoàn tất):** Baseline YOLO26n trên VisDrone ($mAP_{50} = 57.35\%$).
- **WP2 (Hoàn tất):** Merged YOLO26n trên VisDrone + P-DESTRE Đợt 1 ($mAP_{50} = 58.30\%$ VisDrone / $88.36\%$ P-DESTRE 15vids).
- **WP3 (Hoàn tất):** Phân rã hướng đối tượng (OOP) trong `src/`, Dynamic Discovery trong `track_drone.py`.
- **WP4 (Hoàn tất Huấn luyện & Benchmark Chuẩn hóa):** Huấn luyện YOLO26s (Run 3 - 80 epochs) cho trạm mặt đất:
  - Đánh giá chuẩn hóa cùng điều kiện trên 15 video Val P-DESTRE:
    - `yolo26n_visdrone`: $57.35\%$ VisDrone / $47.61\%$ P-DESTRE (rơi rụng $-9.74\%$ do góc dốc $45^\circ-90^\circ$).
    - `yolo26n_merged`: $58.30\%$ VisDrone / $88.36\%$ P-DESTRE (giữ vững tri thức, tối ưu Edge).
    - `yolo26s_merged`: $61.47\%$ VisDrone / $66.80\%$ P-DESTRE (dẫn đầu VisDrone $+4.12\%$ mAP, $+7\%$ Recall, tối ưu trạm mặt đất).
  - Trọng số đã nghiệm thu về local: `models/yolo26s_merged/weights/best.pt` (19.4 MB).
- **WP5 (Hoàn tất Thực nghiệm Đo lường Bám vết):** Đo lường đối chứng 4 cấu hình trên video drone chao liệng (`test_drone_square_pdestre.mp4`):
  - `yolo26n_merged` + `bytetrack`: 624 bboxes, 99 IDs, Live FPS: ~68–84 (nhanh nhưng nhảy ID nhiều).
  - `yolo26n_merged` + `botsort`: 806 bboxes (+29%), 66 IDs (-33% ID switches), Live FPS: ~19–27 (ổn định hơn nhờ GMC nhưng chậm hơn gấp 3).
  - `yolo26s_merged` + `bytetrack`: 540 bboxes, 102 IDs, Live FPS: ~52–80.
  - `yolo26s_merged` + `botsort`: 736 bboxes (+36%), 70 IDs (-31.4% ID switches), Live FPS: ~18–27.
- **WP6 (Hoàn tất — Appearance Memory Bank, OCM, Giao thức Target Lock & Tối ưu UI/UX Thị giác):**
  - **Phân tích kế thừa nghiên cứu Tô Văn Hà ([ToVanHa_42001010.md](file:///d:/record_by_me/NCKH/human_tracking_drone/ToVanHa_42001010.md) & [projectpi5(slow)](file:///d:/record_by_me/NCKH/human_tracking_drone/projectpi5(slow)/projectpi5(slow))):**
    - 4 nút thắt của hệ thống cũ: (1) Camera buộc cố định bằng dây rút $\to$ drone nghiêng là camera cắm đất; (2) Chạy CPU Pi 5 đạt 8-10 FPS @ 320x240; (3) Tracker tham lam theo khoảng cách tâm 80px không có Kalman/Re-ID; (4) Đo khoảng cách bằng BBox dễ sai số.
  - **Định hướng Kỹ thuật Chốt:**
    - Không mua NPU Hailo hay Gimbal đắt tiền $\to$ Giữ kiến trúc **Ground Station Streaming** (xử lý trên Laptop RTX GPU có sẵn).
    - Dùng **Servo 1 trục** giá rẻ để bù góc nghiêng camera khi drone tăng tốc.
    - Dùng **LiDAR mini (đồng trục lens camera)** thay vì BBox để đo khoảng cách chuẩn xác.
    - Xây dựng **Appearance Memory Bank** cho toàn bộ ID: trích xuất vector đặc trưng không gian màu HSV 3 vùng, so khớp Cosine Similarity với template lịch sử để tái kết nối vết (Re-ID) khi đối tượng bị mất dấu tạm thời.
    - Xây dựng giao thức mô phỏng **Target Lock & ACK**:
      - Giai đoạn tìm kiếm: Bật Memory cho toàn bộ ID.
      - Giai đoạn khóa mục tiêu: Gửi gói tin `LOCK_TARGET` $\to$ Pi5 trả về `ACK` $\to$ Chuyển sang cơ chế Dedicated Memory cho target.
      - Giai đoạn bán tự hành & mất dấu: Cho phép phi công can thiệp lái; nếu mất dấu mục tiêu do đổi góc nhìn $\to$ Drone xoay $360^\circ$ quét tìm kiếm dựa trên Memory Bank; nếu hết thời gian chờ không tìm thấy $\to$ Giữ vị trí (Hover).
  - **WP6.1 (Kiểm chứng Lý thuyết & Nâng cấp Thuật toán Tracking):**
    - Nghiên cứu cơ chế Camera Motion Compensation (CMC) và nhận diện giới hạn: Chi phí tính toán CPU cao ($25-40\text{ ms}$), kém ổn định khi nền chuyển động phẳng hoặc thiếu texture, dễ gán nhầm điểm đặc trưng vào người di chuyển.
    - Đề xuất và hiện thực hóa **Observation-Centric Momentum (OCM)**: Tính toán vector vận tốc thực nghiệm từ lịch sử quan sát của từng tracklet, dự phóng tâm đón đầu $\mathbf{p}_{\text{pred}} = \mathbf{p}_t + \mathbf{v}_t \cdot \Delta t$, kết hợp Hungarian Cosine Matching mà không tốn chi phí ma trận Affine nền.
    - Bộ trích xuất ngoại hình thích ứng **Aspect-Ratio Aware Adaptive Slicing**: Tự động chuyển đổi giữa phân rã 3 dải dọc khi $h/w \ge 1.65$ (góc nghiêng $10^\circ-50^\circ$) và 3 vành elip đồng tâm (Concentric Radial Rings) khi $h/w < 1.65$ (góc dốc $50^\circ-90^\circ$).
  - **WP6.2 (Dọn dẹp Toàn bộ Dữ liệu Cũ & Chuẩn hóa Kho 5 Video Flycam Chất lượng cao):**
    - Loại bỏ triệt để toàn bộ các video góc nhìn đỉnh đầu $90^\circ$ Nadir hoặc người vi hạt cự ly quá xa.
    - Dọn sạch kho video và kết quả cũ trong `data/` và `runs/track/`.
    - Thiết lập bộ dữ liệu thực nghiệm chuẩn gồm **5 video flycam chất lượng cao** ($50-60\text{ FPS}$) với góc nghiêng thực tế $27.1^\circ - 58.9^\circ$, người nhìn rõ toàn thân ($h/w = 1.91 - 2.84 \ge 1.65$).
  - **WP6.3 (Tối ưu Trải nghiệm Trực quan UI/UX & Logic Định danh):**
    - Gán ID tuần tự chuẩn xác: Reset bộ đếm và ánh xạ cục bộ từ `#1, #2, #3, ...` cho mỗi video mới, loại bỏ hiện tượng ID nhảy vọt lên hàng chục nghìn.
    - Tinh gọn bảng Telemetry HUD: Giảm kích thước từ $380 \times 165\text{ px}$ xuống **$220 \times 84\text{ px}$** (giảm hơn $60\%$ kích thước), nền bán trong suốt `alpha=0.70`, viền mảnh cyan `0, 180, 220`, chữ sắc nét crisp.
    - Tinh chỉnh nhãn BBox: Giảm cỡ chữ từ $0.48$ xuống **$0.36$**, viền ôm sát vừa vặn, không che khuất đối tượng; mặc định chạy chế độ `--no-trail` cho khung hình quan sát của phi công thông thoáng.
  - **WP6.4 (Thực nghiệm Đối chứng Định lượng 5 Video Flycam — YOLO26n Merged + ByteTrack):**
    - Đánh giá trên toàn bộ **3.715 frames** và **126.258 detections**:
      * Baseline ByteTrack Gốc: **2.183 Unique IDs**.
      * ByteTrack + Adaptive Memory Bank + OCM: **1.322 Unique IDs** (giảm **-39.44%** hiện tượng phân mảnh quỹ đạo).
      * Số lượt Re-ID khôi phục danh tính thành công: **4.185 lượt**.
      * Tốc độ xử lý: Trung bình **45.3 FPS** trên GPU (độ trễ suy luận YOLO26n: **9.26 – 13.31 ms**).
      * Báo cáo định lượng đã lưu: [runs/benchmark_5_hq_oblique_report.csv](file:///d:/record_by_me/NCKH/human_tracking_drone/runs/benchmark_5_hq_oblique_report.csv).
      * Toàn bộ 5 video kết quả bám vết chuẩn mực đã xuất tại: [runs/track/yolo26n_merged_bytetrack_hq_*.mp4](file:///d:/record_by_me/NCKH/human_tracking_drone/runs/track/).

---

### Trạng thái Kỹ thuật:
- **Git:** Đã đồng bộ nhánh `main`, mã nguồn OOP module hóa đồng bộ (`src/features.py`, `src/memory.py`, `src/tracker.py`, `src/visualizer.py`, `src/pipeline.py`, `track_drone.py`).
- **Dữ liệu Video Chuẩn:** Kho 5 video flycam góc nghiêng chất lượng cao trong `data/` (`gettyimages-*.mp4`).
- **Thư mục Kết quả (`runs/`):** Báo cáo định lượng `runs/benchmark_5_hq_oblique_report.csv` và 5 video tracking đối chứng chất lượng cao.
- **Môi trường Thực thi:** Python 3.10, PyTorch 2.6.0+cu124, CUDA True, Torchvision 0.21.0+cu124, OpenCV 4.x.
