"""
Cấu hình hệ thống UAV Human Tracking.
"""

from pathlib import Path

# Model Aliases
MODEL_REGISTRY = {
    "run1": "models/yolo26n_visdrone_filtered_run1/weights/best.pt",
    "run2": "models/yolo26n_merged_run2/weights/best.pt"
}

# Tracker Configurations
TRACKER_REGISTRY = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml"
}

# Default Parameters
DEFAULT_CONF = 0.20
DEFAULT_IOU = 0.50
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "0"
DEFAULT_OUTPUT_DIR = Path("runs/track")
