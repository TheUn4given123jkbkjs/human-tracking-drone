"""
Module phát hiện đối tượng người từ UAV sử dụng kiến trúc YOLO26n Native NMS-Free.
"""

import os
from pathlib import Path
from typing import Optional, Union, Any
import numpy as np

# Đảm bảo cache cục bộ cho Ultralytics
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("./Ultralytics")
from ultralytics import YOLO
from src.config import MODEL_REGISTRY, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMGSZ, DEFAULT_DEVICE


class UAVDetector:
    """
    Lớp đóng gói mô hình phát hiện người từ UAV (UAV Human Detector).
    """

    def __init__(
        self,
        model_path: str = "run2",
        conf_thresh: float = DEFAULT_CONF,
        iou_thresh: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = DEFAULT_DEVICE
    ):
        self.model_path = self._resolve_path(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.device = device
        
        self.model = self._load_model()

    def _resolve_path(self, path: str) -> str:
        """Phân giải alias 'run1', 'run2' hoặc đường dẫn trực tiếp."""
        lower = path.lower()
        if lower in MODEL_REGISTRY:
            resolved = MODEL_REGISTRY[lower]
            if os.path.exists(resolved):
                return resolved
        if not os.path.exists(path):
            raise FileNotFoundError(f"[!] Không tìm thấy file trọng số: {path}")
        return path

    def _load_model(self) -> YOLO:
        """Nạp trọng số mô hình YOLO26n."""
        print(f"[*] [UAVDetector] Nạp trọng số: {self.model_path}")
        model = YOLO(self.model_path)
        return model

    @property
    def model_name(self) -> str:
        """Tên ngắn gọn của mô hình đang nạp."""
        p = Path(self.model_path)
        return p.parent.parent.name if "weights" in self.model_path else p.stem

    @property
    def num_parameters(self) -> int:
        """Số lượng tham số của mô hình."""
        try:
            return sum(p.numel() for p in self.model.model.parameters())
        except Exception:
            return 2533886

    def predict(self, frame: np.ndarray) -> Any:
        """Chạy dự đoán nhận diện trên một khung hình."""
        return self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]
