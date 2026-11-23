"""
Module điều phối thuật toán theo dõi đa đối tượng (Multi-Object Tracking - MOT) cho UAV.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from src.config import TRACKER_REGISTRY
from src.detector import UAVDetector


class UAVTracker:
    """
    Lớp điều phối thuật toán theo dõi đa đối tượng (ByteTrack / BoT-SORT).
    """

    def __init__(
        self,
        detector: UAVDetector,
        tracker_type: str = "bytetrack"
    ):
        self.detector = detector
        self.tracker_type = tracker_type.lower()
        if self.tracker_type not in TRACKER_REGISTRY:
            raise ValueError(f"[!] Thuật toán '{tracker_type}' không hỗ trợ. Lựa chọn: {list(TRACKER_REGISTRY.keys())}")
        
        self.tracker_cfg = TRACKER_REGISTRY[self.tracker_type]
        print(f"[*] [UAVTracker] Khởi tạo thuật toán: {self.tracker_type.upper()} ({self.tracker_cfg})")

    def track(self, frame: np.ndarray) -> Any:
        """
        Thực thi quy trình theo dõi trên khung hình kế tiếp.
        
        Args:
            frame: Khung hình đầu vào (RGB/BGR).
            
        Returns:
            Đối tượng kết quả từ Ultralytics chứa bounding box và track IDs.
        """
        results = self.detector.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.detector.conf_thresh,
            iou=self.detector.iou_thresh,
            imgsz=self.detector.imgsz,
            device=self.detector.device,
            verbose=False
        )
        return results[0]
