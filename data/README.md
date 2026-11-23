# 📁 Dữ Liệu Thực Nghiệm (Datasets) — UAV Human Tracking

> **Đề tài NCKH:** Nhận diện và theo dõi người bằng máy bay không người lái (UAV Human Detection & Tracking)  
> **Lưu trữ tập trung Google Drive:** [Thư mục human_tracking_drone](https://drive.google.com/drive/folders/1khzo4EQoVZZ5bGfdXeUwjeBXHG4KAJyB)

---

## 1. TỔNG QUAN TÀI NGUYÊN TRÊN GOOGLE DRIVE

Hệ thống dữ liệu phục vụ nghiên cứu và kiểm thử được quản lý tập trung trong thư mục `data/` trên Google Drive:

| Thư mục / Tệp trên Drive | Định dạng | Quy mô | Mô tả & Nội dung |
| :--- | :---: | :---: | :--- |
| **`visdrone_filtered_yolo.zip`** | `.zip` | ~1.46 GB | Gói dữ liệu VisDrone sạch (ảnh + nhãn YOLO + splits) |
| **`visdrone_processed/`** | Thư mục | — | Chứa shortcut file nén VisDrone và tệp chia tập `visdrone_splits.zip` |
| **`pdestre_processed/`** | Thư mục | ~2.5 GB | 75 gói zip sequence video trích xuất 2 FPS từ P-DESTRE 4K |
| **`test_videos/`** | Thư mục | ~90 MB | 7 video flycam benchmark thực nghiệm thuật toán bám vết |
| **`raw/`** | Thư mục | ~40 GB | Chứa toàn bộ dữ liệu gốc VisDrone-DET và P-DESTRE (`dataset.tar`) |

---

## 2. ĐẶC TẢ HAI BỘ DỮ LIỆU CHÍNH

### 2.1. VisDrone-DET (Giai đoạn 1 — Baseline Đô Thị Góc Nghiêng)
- **Quy mô mẫu:**
  - **3.739 ảnh sạch** (3.299 ảnh Train, 440 ảnh Val đối chứng độc lập).
  - **39.857 bounding box người** đã hợp nhất về nhãn duy nhất `class 0: person`.
- **Độ cao & Góc nhìn phục vụ:**
  - **Độ cao hành trình 5 – 20m (Trường hợp B):** 34.558 boxes ($32\text{px} \le h < 120\text{px}$), là vùng hoạt động chủ đạo của drone tuần tra.
  - **Độ cao cực thấp < 5m (Trường hợp A):** 5.299 boxes ($h \ge 120\text{px}$), phục vụ pha tiếp cận mục tiêu.
  - **Góc nhìn (Pitch Angle):** Góc nghiêng **$15^\circ - 45^\circ$**, bao phủ đô thị đông đúc, đường giao thông và bãi đỗ xe.
- **Tiền xử lý đặc thù:**
  - Mask đen tuyền 81.540 vùng *ignored* và người ngoài tầm hiệu dụng ($>20\text{m}$, $h < 32\text{px}$) trên tập Train để khử gradient nhiễu, bảo toàn độ chính xác cho hàm loss.

### 2.2. P-DESTRE (Giai đoạn 2 — Mở Rộng Góc Dốc & Đỉnh Đầu)
- **Quy mô mẫu:**
  - **75 chuỗi video 4K UHD** ($3840 \times 2160$ @ 30 FPS).
  - Trích xuất cố định tại **$2\text{ FPS}$** (downsampling $1/15$) để khử hiện tượng tự tương quan chuỗi thời gian.
  - Phân chia độc lập theo cấp độ chuỗi video (**60 video train, 15 video val**) nhằm loại bỏ hoàn toàn hiện tượng rò rỉ dữ liệu (*data leakage*).
- **Độ cao & Góc nhìn phục vụ:**
  - **Độ cao hoạt động:** **5.5 – 6.7m**.
  - **Góc nhìn (Pitch Angle):** Góc dốc cực lớn từ **$45^\circ$ đến $90^\circ$ (Bird's-Eye View)**.
  - **Mục tiêu bài toán:** Giải quyết triệt để điểm mù góc nhìn đỉnh đầu/vai khi camera drone chúc thẳng xuống quảng trường, công viên, lối đi bộ.

### 2.3. Bộ 7 Video Flycam Kiểm Thử Thực Tế (`test_videos/`)
Phục vụ so sánh thực nghiệm hiệu năng giữa **ByteTrack** và **BoT-SORT**:
- Độ cao: **10 – 20m**; Góc nghiêng: **$30^\circ - 50^\circ$**.
- Đa dạng mật độ: Từ nhóm người di chuyển phân tán đến đám đông phức tạp ($>45$ người/frame) với hiện tượng che khuất liên tục.

---

## 3. HƯỚNG DẪN ĐỒNG BỘ DỮ LIỆU TỪ GOOGLE DRIVE

### 3.1. Đồng Bộ Nhanh Bằng `gdrive` CLI
Sử dụng công cụ `gdrive` để tải nhanh các gói dữ liệu từ thư mục `data/` trên Drive:

```powershell
# 1. Tìm và tải file nén VisDrone Processed
gdrive readonly download "visdrone_filtered_yolo.zip" --out "data/processed/visdrone_filtered_yolo.zip"

# 2. Tải toàn bộ video test về thư mục data local
gdrive readonly download "test_videos" --out "data/"
```

### 3.2. Đồng Bộ Bằng Python Script
Sử dụng Python để tự động tải và giải nén dữ liệu đã xử lý phục vụ huấn luyện:

```python
import zipfile
from pathlib import Path
import gdown

# Đường dẫn tải gói dữ liệu VisDrone Processed từ Google Drive
url = "https://drive.google.com/uc?id=visdrone_filtered_yolo"
output_zip = Path("data/processed/visdrone_filtered_yolo.zip")
output_zip.parent.mkdir(parents=True, exist_ok=True)

# Tải và giải nén
gdown.download(url, str(output_zip), quiet=False)
with zipfile.ZipFile(output_zip, "r") as zf:
    zf.extractall("data/processed/visdrone")
print("✅ Hoàn tất đồng bộ dữ liệu VisDrone Processed!")
```
