"""
Cấu hình hệ thống UAV Human Tracking.
Tuân thủ nguyên tắc Dynamic Discovery (quét động mô hình từ thư mục models/).
"""

from pathlib import Path

# Thư mục lưu trữ mô hình và kết quả
MODELS_DIR = Path("models")
DEFAULT_OUTPUT_DIR = Path("runs/track")

# Mô hình mặc định (Mô hình Ground Station tốt nhất hiện tại)
DEFAULT_MODEL = "yolo26s_merged"

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
