"""
Module điều phối thuật toán theo dõi đa đối tượng (Multi-Object Tracking - MOT) cho UAV.
Tích hợp nâng cấp: Appearance Memory Bank chống phân mảnh quỹ đạo và nhảy ID.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from src.config import (
    TRACKER_REGISTRY,
    DEFAULT_ADAPTIVE_EXTRACTION,
    DEFAULT_FEATURE_UPDATE_INTERVAL
)
from src.detector import UAVDetector
from src.memory import AppearanceMemoryBank


class UAVTracker:
    """
    Lớp điều phối thuật toán theo dõi đa đối tượng (ByteTrack / BoT-SORT)
    kết hợp tầng Appearance Memory Bank để tái nhận diện (Re-ID) mục tiêu.
    """

    def __init__(
        self,
        detector: UAVDetector,
        tracker_type: str = "bytetrack",
        use_memory: bool = True,
        similarity_thresh: float = 0.62,
        max_lost_frames: int = 60,
        max_spatial_distance: float = 450.0,
        memory_mode: str = "all",
        adaptive_extraction: bool = DEFAULT_ADAPTIVE_EXTRACTION,
        feature_update_interval: int = DEFAULT_FEATURE_UPDATE_INTERVAL
    ):
        self.detector = detector
        self.tracker_type = tracker_type.lower()
        if self.tracker_type not in TRACKER_REGISTRY:
            raise ValueError(f"[!] Thuật toán '{tracker_type}' không hỗ trợ. Lựa chọn: {list(TRACKER_REGISTRY.keys())}")
        
        self.tracker_cfg = TRACKER_REGISTRY[self.tracker_type]
        self.use_memory = use_memory
        self.frame_idx = 0
        self.baseline_id_map: Dict[int, int] = {}
        self.next_baseline_id: int = 1

        # Khởi tạo Memory Bank nếu được kích hoạt
        if self.use_memory:
            self.memory_bank = AppearanceMemoryBank(
                similarity_thresh=similarity_thresh,
                max_lost_frames=max_lost_frames,
                max_spatial_distance=max_spatial_distance,
                mode=memory_mode,
                adaptive_extraction=adaptive_extraction,
                feature_update_interval=feature_update_interval
            )
            print(f"[*] [UAVTracker] Khởi tạo thuật toán: {self.tracker_type.upper()} ({self.tracker_cfg}) + [APPEARANCE MEMORY BANK ON]")
        else:
            self.memory_bank = None
            print(f"[*] [UAVTracker] Khởi tạo thuật toán: {self.tracker_type.upper()} ({self.tracker_cfg}) [MEMORY OFF]")

    def track(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> Any:
        """
        Thực thi quy trình theo dõi trên khung hình kế tiếp.
        
        Args:
            frame: Khung hình đầu vào (RGB/BGR).
            frame_idx: Số thứ tự frame (tùy chọn, tự động tăng nếu không truyền).
            
        Returns:
            Đối tượng kết quả từ Ultralytics chứa bounding box và track IDs đã được chuẩn hóa qua Memory Bank.
        """
        if frame_idx is not None:
            self.frame_idx = frame_idx
        else:
            self.frame_idx += 1

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
        res = results[0]

        # Áp dụng cơ chế Appearance Memory Bank nếu có kết quả bám vết
        if self.use_memory and res.boxes is not None and res.boxes.id is not None and len(res.boxes.id) > 0:
            raw_ids = res.boxes.id.int().cpu().tolist()
            xyxys = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else [0.0] * len(raw_ids)

            raw_detections = []
            for raw_id, xyxy, conf in zip(raw_ids, xyxys, confs):
                x1, y1, x2, y2 = map(int, xyxy)
                raw_detections.append({
                    "raw_id": raw_id,
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(conf)
                })

            # Xử lý qua Memory Bank
            processed = self.memory_bank.process_frame(frame, self.frame_idx, raw_detections)

            # Cập nhật trực tiếp canonical ID vào cấu trúc tensor của Ultralytics Boxes
            # res.boxes.data[:, 4] là cột track_id trong định dạng chuẩn [x1, y1, x2, y2, track_id, conf, cls]
            canonical_ids = [p["track_id"] for p in processed]
            if len(canonical_ids) == len(res.boxes.data):
                new_data = res.boxes.data.clone()
                new_data[:, 4] = torch.tensor(canonical_ids, dtype=new_data.dtype, device=new_data.device)
                res.boxes.data = new_data

            # Đính kèm metadata mở rộng vào đối tượng kết quả
            res.memory_processed = processed
            res.reacquired_this_frame = [p for p in processed if p.get("reacquired", False)]
        else:
            # Chế độ Baseline: Ánh xạ tuần tự các ID từ 1 cho từng video
            if res.boxes is not None and res.boxes.id is not None and len(res.boxes.id) > 0:
                raw_ids = res.boxes.id.int().cpu().tolist()
                clean_ids = []
                for rid in raw_ids:
                    if rid not in self.baseline_id_map:
                        self.baseline_id_map[rid] = self.next_baseline_id
                        self.next_baseline_id += 1
                    clean_ids.append(self.baseline_id_map[rid])
                if len(clean_ids) == len(res.boxes.data):
                    new_data = res.boxes.data.clone()
                    new_data[:, 4] = torch.tensor(clean_ids, dtype=new_data.dtype, device=new_data.device)
                    res.boxes.data = new_data

            res.memory_processed = []
            res.reacquired_this_frame = []

        return res

    def reset(self):
        """Khởi động lại trạng thái tracker cho video mới (bắt đầu ID từ 1)."""
        self.frame_idx = 0
        self.baseline_id_map.clear()
        self.next_baseline_id = 1
        try:
            from ultralytics.trackers.basetrack import BaseTrack
            BaseTrack._count = 0
        except Exception:
            pass
        if self.memory_bank is not None:
            self.memory_bank.reset()
