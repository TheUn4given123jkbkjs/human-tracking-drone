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
- **WP6 (Đang thực hiện — Appearance Memory Bank & Target Lock Protocol):**
  - **Phân tích kế thừa nghiên cứu Tô Văn Hà ([ToVanHa_42001010.md](file:///d:/record_by_me/NCKH/human_tracking_drone/ToVanHa_42001010.md) & [projectpi5(slow)](file:///d:/record_by_me/NCKH/human_tracking_drone/projectpi5(slow)/projectpi5(slow))):**
    - 4 nút thắt của hệ thống cũ: (1) Camera buộc cố định bằng dây rút $\to$ drone nghiêng là camera cắm đất; (2) Chạy CPU Pi 5 đạt 8-10 FPS @ 320x240; (3) Tracker tham lam theo khoảng cách tâm 80px không có Kalman/Re-ID; (4) Đo khoảng cách bằng BBox dễ sai số.
  - **Định hướng Kỹ thuật Chốt:**
    - Không mua NPU Hailo hay Gimbal đắt tiền $\to$ Giữ kiến trúc **Ground Station Streaming** (xử lý trên Laptop RTX GPU có sẵn).
    - Dùng **Servo 1 trục** giá rẻ để bù góc nghiêng camera khi drone tăng tốc.
    - Dùng **LiDAR mini (đồng trục lens camera)** thay vì BBox để đo khoảng cách chuẩn xác.
    - Xây dựng **Appearance Memory Bank** cho toàn bộ ID: trích xuất vector đặc trưng không gian màu HSV 3 vùng (đầu/thân/chân), so khớp Cosine Similarity với template lịch sử để tái kết nối vết (Re-ID) khi đối tượng bị mất dấu tạm thời.
    - Xây dựng giao thức mô phỏng **Target Lock & ACK**:
      - Giai đoạn tìm kiếm: Bật Memory cho toàn bộ ID.
      - Giai đoạn khóa mục tiêu: Gửi gói tin `LOCK_TARGET` $\to$ Pi5 trả về `ACK` $\to$ Chuyển sang cơ chế Dedicated Memory cho target.
      - Giai đoạn bán tự hành & mất dấu: Cho phép phi công can thiệp lái; nếu mất dấu mục tiêu do đổi góc nhìn $\to$ Drone xoay $360^\circ$ quét tìm kiếm dựa trên Memory Bank; nếu hết thời gian chờ không tìm thấy $\to$ Giữ vị trí (Hover).

---

### Trạng thái Kỹ thuật:
- **Git:** Đã đồng bộ nhánh `main`, `context.md` cũ đã được loại bỏ để tập trung vào `context_checked.md`.
- **Môi trường Thực thi:** Python 3.10, PyTorch 2.6.0+cu124, CUDA True, Torchvision 0.21.0+cu124, OpenCV 4.x.
