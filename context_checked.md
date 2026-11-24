# Context Checkpoint — Nghiên cứu Khoa học
> **Cập nhật ngày:** 24/11/2026 (Sau khi hoàn tất Run 3 — YOLO26s)

- **Tên Đề tài / Mục tiêu:** Nhận diện và theo dõi người bằng máy bay không người lái (UAV Human Detection & Tracking) sử dụng kiến trúc YOLO26 Native NMS-Free kết hợp ByteTrack & BoT-SORT.
- **Vấn đề Khoa học & Giả thuyết ($H_0/H_1$):**
  - $H_0$: Các mô hình thị giác truyền thống bị nghẽn độ trễ do tầng NMS khi mật độ đông, đồng thời bị mù góc nhìn khi camera drone chúc dốc $45^\circ - 90^\circ$.
  - $H_1$: Kiến trúc YOLO26 Native NMS-Free kết hợp dữ liệu hợp nhất (VisDrone + P-DESTRE) duy trì độ trễ phẳng ($< 1.0\text{ ms}$) và triệt tiêu điểm mù đỉnh đầu/vai.
- **Gói Công việc Hiện tại (Work Package):**
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
    - `yolo26n_merged` + `bytetrack`: 624 bboxes, 99 IDs, Live FPS: ~68–84.
    - `yolo26n_merged` + `botsort`: 806 bboxes (+29%), 66 IDs (-33% ID switches), Live FPS: ~19–27.
    - `yolo26s_merged` + `bytetrack`: 540 bboxes, 102 IDs, Live FPS: ~52–80.
    - `yolo26s_merged` + `botsort`: 736 bboxes (+36%), 70 IDs (-31.4% ID switches), Live FPS: ~18–27.
- **Trạng thái Dữ liệu & Khóa nhãn:** Đã cô lập và đồng bộ đầy đủ trên Google Drive & GitHub.
- **Các Kết quả & Quyết định Đã Nghiệm thu:**
  - Kiến trúc hệ thống: Phi công lái drone thủ công bằng tay cầm RC (an toàn bay 100%), video truyền về máy trạm GPU chạy mô hình AI (YOLO26s + ByteTrack/BoT-SORT).
  - Sự đánh đổi thực nghiệm: BoT-SORT giảm 31–33% ID Switches nhờ cơ chế bù rung lắc camera GMC nhưng tốc độ (~20–27 FPS) thấp hơn ByteTrack (~60–80 FPS).
- **Nhiệm vụ Tiếp theo:**
  - [x] Chạy benchmark thực nghiệm MOT trên video drone: đo ID Switches, FPS cho cả ByteTrack và BoT-SORT.
  - [x] Cập nhật bảng đối chứng và nhận định khoa học vào `README.md`.
  - [ ] Commit mã nguồn sạch lên nhánh `main` của repository GitHub.
