# Context Checkpoint — Nghiên cứu Khoa học
> **Cập nhật ngày:** 26/11/2026 — **Nghiệm thu Milestone v1.0 (Hoàn thành WP1 – WP6), Đang thực hiện WP7 & Khởi tạo WP8**

- **Tên Đề tài / Mục tiêu:** Nhận diện và theo dõi người bằng máy bay không người lái (UAV Human Detection & Tracking) thời gian thực sử dụng kiến trúc YOLO26 Native NMS-Free kết hợp ByteTrack & BoT-SORT cải tiến thêm tầng Appearance Memory Bank và hệ thống truyền phát video mặt đất (Ground Station Streaming).
- **Vấn đề Khoa học & Giả thuyết ($H_0/H_1$):**
  - $H_0$: Các mô hình thị giác truyền thống bị nghẽn độ trễ do tầng NMS khi mật độ đông, mất dấu khi camera drone chao liệng, và các thuật toán tracking thuần hình học (IoU/Kalman) bị đứt gãy, nhảy ID liên tục khi đối tượng bị che khuất tạm thời hoặc đổi góc nhìn đột ngột. Hệ thống truyền phát Wi-Fi drone dễ crash khi mất gói hoặc rớt sóng tạm thời.
  - $H_1$: Kiến trúc YOLO26 Native NMS-Free kết hợp dữ liệu hợp nhất (VisDrone + P-DESTRE) duy trì độ trễ phẳng ($< 1.0\text{ ms}$), kết hợp tầng **Appearance Memory Bank** (lưu trữ vector đặc trưng màu/không gian đa vùng và cosine matching) giúp tái nhận diện (Re-ID) và khôi phục ID bị mất dấu, giảm thiểu tối đa hiện tượng phân mảnh quỹ đạo (ID switches). Hệ thống thu nhận đa nguồn với cơ chế **Auto-Reconnect & Frame Hold** đảm bảo trạm điều khiển mặt đất (GCS) duy trì liên tục và phục hồi tức thì khi có sóng.

---

### Các Gói Công việc Đã Hoàn tất (Work Packages WP1 – WP6):

- **WP1 (Hoàn tất):** Baseline YOLO26n trên VisDrone ($mAP_{50} = 57.35\%$).
- **WP2 (Hoàn tất):** Merged YOLO26n trên VisDrone + P-DESTRE Đợt 1 ($mAP_{50} = 58.30\%$ VisDrone / $88.36\%$ P-DESTRE 15vids).
- **WP3 (Hoàn tất):** Phân rã hướng đối tượng (OOP) trong `src/`, Dynamic Discovery trong `track_drone.py`.
- **WP4 (Hoàn tất Huấn luyện & Benchmark Chuẩn hóa):** Huấn luyện YOLO26s (Run 3 - 80 epochs) cho trạm mặt đất:
  - Đánh giá chuẩn hóa cùng điều kiện trên 15 video Val P-DESTRE:
    - `yolo26n_visdrone`: $57.35\%$ VisDrone / $47.61\%$ P-DESTRE (rơi rụng $-9.74\%$ do góc dốc $45^\circ-90^\circ$).
    - `yolo26n_merged`: $58.30\%$ VisDrone / $88.36\%$ P-DESTRE (giữ vững tri thức, tối ưu Edge/Onboard).
    - `yolo26s_merged`: $61.47\%$ VisDrone / $66.80\%$ P-DESTRE (dẫn đầu VisDrone $+4.12\%$ mAP, $+7\%$ Recall, tối ưu trạm mặt đất).
  - Trọng số đã nghiệm thu về local: `models/yolo26s_merged/weights/best.pt` (19.4 MB) và `models/yolo26n_merged/weights/best.pt` (5.2 MB).
- **WP5 (Hoàn tất Thực nghiệm Đo lường Bám vết):** Đo lường đối chứng 4 cấu hình trên video drone chao liệng (`test_drone_square_pdestre.mp4`):
  - `yolo26n_merged` + `bytetrack`: 624 bboxes, 99 IDs, Live FPS: ~68–84 (nhanh nhưng nhảy ID nhiều).
  - `yolo26n_merged` + `botsort`: 806 bboxes (+29%), 66 IDs (-33% ID switches), Live FPS: ~19–27.
  - `yolo26s_merged` + `bytetrack`: 540 bboxes, 102 IDs, Live FPS: ~52–80.
  - `yolo26s_merged` + `botsort`: 736 bboxes (+36%), 70 IDs (-31.4% ID switches), Live FPS: ~18–27.
- **WP6 (Hoàn tất — Appearance Memory Bank & Observation-Centric Momentum):**
  - **Kế thừa & Tối ưu hóa:** Bỏ phương án NPU Hailo/Gimbal đắt tiền, tập trung kiến trúc Ground Station Streaming; tích hợp Servo 1 trục bù góc nghiêng camera và LiDAR mini đồng trục.
  - **Observation-Centric Momentum (OCM):** Tính vector vận tốc thực nghiệm từ lịch sử quan sát của từng tracklet, dự phóng tâm đón đầu $\mathbf{p}_{\text{pred}} = \mathbf{p}_t + \mathbf{v}_t \cdot \Delta t$, kết hợp Hungarian Cosine Matching mà không tốn chi phí ma trận Affine nền.
  - **Aspect-Ratio Aware Adaptive Slicing:** Tự động chuyển đổi giữa phân rã 3 dải dọc khi $h/w \ge 1.65$ (góc nghiêng $10^\circ-50^\circ$) và 3 vành elip đồng tâm (Concentric Radial Rings) khi $h/w < 1.65$ (góc dốc $50^\circ-90^\circ$).
  - **Thực nghiệm Định lượng 5 Video Flycam Chuẩn:**
    - Baseline ByteTrack Gốc: 2.183 Unique IDs.
    - ByteTrack + Adaptive Memory Bank + OCM: 1.322 Unique IDs (giảm **-39.44%** hiện tượng phân mảnh quỹ đạo).
    - Khôi phục danh tính thành công: **4.185 lượt Re-ID**.
    - Tốc độ xử lý trung bình: **45.3 FPS** trên GPU.

---

### Các Gói Công việc Đang Triển khai & Kế hoạch Tiếp theo (WP7 & WP8):

- **WP7 (Đang thực hiện — Liên kết Wi-Fi Drone - GCS, Truyền phát Video Downlink & Bộ Thu Đa Nguồn Chống Rớt Sóng):**
  - **Trạng thái:** `[ ] Đang thực hiện (Hoàn thành tầng mô phỏng & kiến trúc phần mềm máy trạm; CHƯA KẾT NỐI VÀ KIỂM THỬ TRÊN DRONE PHẦN CỨNG THỰC TẾ)`.
  - **Đã hoàn thành (Software Stack & Simulation Test):**
    - **Bộ thu nhận video đa nguồn máy trạm (`src/stream_receiver.py`):** Thiết kế Factory Pattern hỗ trợ File, Camera DirectShow và luồng mạng RTSP/UDP. Cài đặt Worker Thread bất đồng bộ liên tục đọc frame mới nhất, triệt tiêu tích lũy trễ (Queue-free buffer).
    - **Chống ngắt luồng khi mất frame Wi-Fi (Auto-Reconnect & Frame Hold trong `src/pipeline.py`):** Tự động giải phóng socket và thử kết nối lại ngầm mỗi $1.0\text{ s}$ khi mất liên tiếp $\ge 25$ frame; giữ khung hình cuối, làm tối 30% kèm cảnh báo đỏ `[!] SIGNAL LOST - WAITING RECONNECT...`; đóng băng Kalman/Memory Bank và tự bù trừ dòng thời gian (`start_time += elapsed_lost`) để khôi phục Re-ID tức thì khi có sóng lại. Cho phép phi công thoát chủ động bằng `Q`/`ESC`.
    - **Bộ mô phỏng kênh truyền & phát luồng RTSP (`src/stream_simulator.py`):** Giả lập kênh truyền phát video H.264 từ camera drone, điều tiết bitrate/FPS và mô phỏng suy hao tín hiệu / rớt gói phục vụ kiểm thử phần mềm độc lập.
    - **Tối ưu hiển thị & lưu trữ:** Direct RAM View 0ms latency; Bộ ghi video H.264 P2 Visual Lossless CRF 18 qua PyAV.
    - **Adaptive Event-Triggered Extraction (`src/memory.py`):** Tự động kích hoạt trích xuất 100% khi tương tác gần, tiết kiệm $21.7\%$ CPU trên đám đông 180 người.
    - **Chuẩn hóa Dự án & Kiểm thử:** Đã tạo `requirements.txt` chuẩn hóa; tinh gọn `README.md` (159 dòng); kiểm thử hồi quy 4/4 bài test pass 100% khớp với v1.0.
  - **Các đầu việc còn lại cần triển khai thực tế trên phần cứng (Hardware Deployment & Flight Test - PENDING):**
    - [ ] **Cấu hình liên kết mạng Wi-Fi thực tế Drone - GCS:** Thiết lập kết nối mạng vô tuyến thực tế (Access Point / Ad-hoc giữa Drone và Máy tính trạm), gán IP tĩnh, thiết lập dải băng tần 2.4GHz / 5.8GHz và kiểm tra suy hao gói tin theo khoảng cách.
    - [ ] **Mô-đun truyền phát từ Camera Drone thật (Onboard Video Streaming Script):** Viết và cài đặt script lấy luồng video trực tiếp từ camera phần cứng trên drone (Raspberry Pi Camera / ESP32-CAM / Camera USB gimbal) nén H.264 và đẩy luồng RTSP/UDP về máy trạm.
    - [ ] **Thực nghiệm bay thực địa (Field Flight Test):** Kết nối trực tiếp máy trạm với drone đang bay, thu nhận luồng video thực tế từ trên không và kiểm tra độ trễ truyền dẫn (End-to-End Latency).
    - [ ] **Kiểm chứng cơ chế Auto-Reconnect ngoài trời:** Bay drone ra ngoài vùng phủ sóng Wi-Fi hoặc tắt bật nguồn phát trên drone để kiểm tra khả năng tự kết nối lại và khôi phục ID mục tiêu trong điều kiện thực tế.
- **WP8 (Kế hoạch Mới — Giao thức Khóa Mục tiêu Bán tự hành & Điều khiển Tương tác GCS - Drone):**
  - **Trạng thái:** `[ ] Đang lập kế hoạch / Chưa thực hiện (Pending - Tách riêng theo yêu cầu thực nghiệm)`.
  - **Mục tiêu & Cơ sở Khoa học:** Thiết kế giao thức điều khiển C2 hai chiều (Command & Control Uplink / Downlink) độc lập với luồng video downlink, áp dụng cơ chế bắt tay xác thực ACK bắt buộc, máy trạng thái hữu hạn (FSM 4 trạng thái) và tương tác người-máy (Human-in-the-loop) hỗ trợ phi công khóa mục tiêu theo dõi.
  - **Phân rã nhiệm vụ chi tiết:**
    - **WP8.1 (Giao thức C2 & Bản tin Xác thực ACK):** Hoàn thiện cấu trúc bản tin `LockTargetCommand(target_id, bbox_coords, timestamp)` (GCS $\to$ Drone) và `LockTargetAck(target_id, status, error_code)` (Drone $\to$ GCS) qua socket mạng UDP/TCP C2 riêng biệt, có cơ chế Timeout Retry.
    - **WP8.2 (Máy trạng thái FSM 4 bước trong `src/target_protocol.py`):** Kiểm định chuyển dịch trạng thái an toàn: `IDLE` (quét toàn cảnh) $\to$ `LOCKING` (chờ xác nhận bắt tay) $\to$ `LOCKED` (khóa cứng mục tiêu) $\to$ `LOST` / `ABORT` (mục tiêu mất dấu hoặc phi công hủy lệnh).
    - **WP8.3 (Giao diện Tương tác Chuột Phi công trên GCS Viewer):** Bắt sự kiện chuột `cv2.setMouseCallback` trên cửa sổ `H264StreamViewer` / `--view`. Khi phi công click vào BBox của một người, hệ thống map tọa độ pixel $(x, y) \to \text{Track ID}$, kích hoạt lệnh gửi khóa.
    - **WP8.4 (Visual Feedback & Feedback Control Loop):** BBox mục tiêu bị khóa chuyển màu Đỏ rực (`[TARGET LOCK #ID]`), HUD hiển thị góc lệch tâm $(\Delta x, \Delta y)$ làm đầu vào cho thuật toán điều khiển bám tâm (Visual Servoing) lái gimbal/drone.

---

### Kết quả Đánh giá Phản biện Khoa học (Audit Findings):
1. **Trọng tâm AI & Tracking (80%):**
   - YOLO26n Merged hoạt động ổn định ở tốc độ cao trên thiết bị biên. Cần hoàn thiện biểu đồ đo kiểm chứng minh độ trễ hậu xử lý phẳng hoàn toàn độc lập với IoU threshold để khẳng định Native NMS-Free.
   - Appearance Memory Bank kết hợp OCM đạt hiệu quả vượt trội trong giảm ID switches (-39.44%). Cần lưu ý giới hạn của biểu đồ màu HSV khi ánh sáng thay đổi đột ngột giữa nắng và bóng râm.
2. **Trọng tâm Ứng dụng Thực nghiệm UAV (20%):**
   - **Liên kết Wi-Fi & Thu nhận Video Downlink (WP7 - Đang thực hiện):** Kiến trúc phần mềm máy trạm (`stream_receiver.py`) và cơ chế Auto-Reconnect / Frame Hold đã kiểm chứng thành công trên mô phỏng. **Tuyệt đối chưa nghiệm thu hoàn tất WP7 vì chưa kết nối với bất kỳ drone phần cứng thực tế nào**. Cần hoàn thiện việc nạp script streaming lên camera drone thật và thực hiện bay thực địa để kiểm tra độ ổn định sóng Wi-Fi ngoài trời.
   - **Giao thức Khóa mục tiêu (Target Lock - WP8 - Chưa thực hiện):** Đã tách thành gói công việc riêng biệt. Cần kết nối module máy trạng thái `src/target_protocol.py` với sự kiện chuột trên giao diện GCS `--view` và thiết lập kênh truyền C2 hai chiều có ACK độc lập với luồng video.
3. **Tính nhất quán & Mã nguồn:**
   - Đã đồng bộ các module chính trong `src/`. Đã đưa hằng số `DEFAULT_SIM_THRESH = 0.60` và `DEFAULT_MAX_LOST_FRAMES = 60` vào `src/config.py` làm nguồn chân lý duy nhất.

---

### Trạng thái Kỹ thuật Hiện tại:
- **Git Branch:** `feature/realtime-streaming-simulator`
- **Mã nguồn lõi (`src/`):**
  - `config.py`: Cấu hình tham số toàn cục.
  - `detector.py`: Đóng gói YOLO26 Native NMS-Free & Dynamic Discovery.
  - `features.py`: Trích xuất đặc trưng ngoại hình thích ứng (Vertical & Concentric Radial).
  - `memory.py`: Quản lý Appearance Memory Bank & OCM.
  - `metrics.py`: Đo lường độ trễ và thống kê thời gian thực.
  - `pipeline.py`: Điều phối luồng xử lý, Auto-Reconnect, Frame Hold & H264VideoWriter.
  - `stream_receiver.py`: Bộ thu nhận video đa nguồn (File, Camera, RTSP với Auto-Reconnect).
  - `stream_simulator.py`: Mô phỏng chất lượng truyền phát và cửa sổ Direct RAM View.
  - `target_protocol.py`: Giao thức khóa mục tiêu và máy trạng thái bán tự hành.
  - `tracker.py`: Tích hợp ByteTrack & BoT-SORT với Memory Bank.
  - `visualizer.py`: Vẽ BBox viền mảnh, vệt quỹ đạo và Telemetry HUD.
- **Entrypoint:** `track_drone.py` (CLI chuẩn hóa).
- **Môi trường Thực thi:** Python 3.10 / 3.13, PyTorch 2.6.0+cu124, CUDA True, PyAV 14.x, OpenCV 4.x.
