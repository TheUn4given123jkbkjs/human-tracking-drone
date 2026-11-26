"""
Cấu hình hệ thống UAV Human Tracking.
Tuân thủ nguyên tắc Dynamic Discovery (quét động mô hình từ thư mục models/).
"""

from pathlib import Path

# Thư mục lưu trữ mô hình và kết quả
MODELS_DIR = Path("models")
DEFAULT_OUTPUT_DIR = Path("runs/track")

# Mô hình mặc định (YOLO26n Nano - Tối ưu thời gian thực >= 30-60 FPS)
DEFAULT_MODEL = "yolo26n_merged"

# Cấu hình Tracker có sẵn
TRACKER_REGISTRY = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml"
}
DEFAULT_TRACKER = "bytetrack"

# Tham số mặc định của hệ thống
DEFAULT_CONF = 0.20
DEFAULT_IOU = 0.50
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "0"

# Cấu hình Tối ưu hóa Bộ nhớ Ngoại hình (Adaptive Event-Triggered Appearance Memory)
DEFAULT_ADAPTIVE_EXTRACTION = True
DEFAULT_FEATURE_UPDATE_INTERVAL = 3
DEFAULT_PROXIMITY_THRESH = 60.0  # Khoảng cách tâm pixel kích hoạt bảo vệ chống nhảy ID
DEFAULT_LOW_CONF_THRESH = 0.35   # Ngưỡng tin cậy kích hoạt bảo vệ
DEFAULT_SIM_THRESH = 0.60        # Ngưỡng Cosine Similarity để Re-ID mục tiêu
DEFAULT_MAX_LOST_FRAMES = 60     # Số frame tối đa lưu vết mục tiêu bị mất dấu
