"""
Module quản lý và tính toán chỉ số hiệu năng thực nghiệm (Tracking Metrics).
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Any
import numpy as np


@dataclass
class TrackingMetrics:
    """Class lưu trữ và phân tích các chỉ số thời gian thực phục vụ NCKH."""
    total_frames: int = 0
    total_time_ms: float = 0.0
    inference_times_ms: List[float] = field(default_factory=list)
    postprocess_times_ms: List[float] = field(default_factory=list)
    unique_track_ids: Set[int] = field(default_factory=set)
    total_detections: int = 0
    active_targets_per_frame: List[int] = field(default_factory=list)

    @property
    def average_fps(self) -> float:
        """Tốc độ trung bình (FPS) toàn chuỗi xử lý."""
        return (self.total_frames / (self.total_time_ms / 1000.0)) if self.total_time_ms > 0 else 0.0

    @property
    def avg_inference_ms(self) -> float:
        """Độ trễ suy luận trung bình của mạng nơ-ron (ms)."""
        return float(np.mean(self.inference_times_ms)) if self.inference_times_ms else 0.0

    @property
    def avg_postprocess_ms(self) -> float:
        """Độ trễ hậu xử lý Native NMS-Free trung bình (ms)."""
        return float(np.mean(self.postprocess_times_ms)) if self.postprocess_times_ms else 0.0

    @property
    def avg_active_targets(self) -> float:
        """Số đối tượng người xuất hiện đồng thời trung bình mỗi frame."""
        return float(np.mean(self.active_targets_per_frame)) if self.active_targets_per_frame else 0.0

    def record_frame(self, inf_ms: float, post_ms: float, active_count: int, ids: List[int]):
        """Ghi nhận dữ liệu của 1 khung hình."""
        self.total_frames += 1
        self.inference_times_ms.append(inf_ms)
        self.postprocess_times_ms.append(post_ms)
        self.active_targets_per_frame.append(active_count)
        self.total_detections += active_count
        for tid in ids:
            self.unique_track_ids.add(tid)

    def to_dict(self) -> Dict[str, Any]:
        """Xuất thông số thành dictionary."""
        return {
            "total_frames": self.total_frames,
            "average_fps": round(self.average_fps, 2),
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "avg_postprocess_ms": round(self.avg_postprocess_ms, 2),
            "total_detections": self.total_detections,
            "unique_track_ids": len(self.unique_track_ids),
            "avg_active_targets": round(self.avg_active_targets, 2)
        }
