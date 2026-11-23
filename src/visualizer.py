"""
Module trực quan hóa kết quả bám vết (Bounding box, Trajectory trail, UAV Telemetry HUD).
"""

from typing import Dict, List, Tuple, Any
import cv2
import numpy as np


class TrackingVisualizer:
    """
    Lớp trực quan hóa chuyên dụng cho bám vết đối tượng người từ UAV.
    """

    def __init__(self, max_trail_len: int = 35, show_trail: bool = True):
        self.max_trail_len = max_trail_len
        self.show_trail = show_trail
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Tạo màu sắc riêng biệt, cố định và nổi bật cho từng ID."""
        color_r = (track_id * 67) % 255
        color_g = (track_id * 131) % 255
        color_b = (track_id * 193) % 255
        return (color_b, color_g, color_r)

    def draw_hud(
        self,
        frame: np.ndarray,
        model_name: str,
        tracker_name: str,
        fps: float,
        inference_ms: float,
        postprocess_ms: float,
        active_count: int,
        unique_total: int
    ) -> np.ndarray:
        """Vẽ bảng đo lường Telemetry HUD góc trên bên trái khung hình drone."""
        hud_w, hud_h = 370, 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (15, 15), (15 + hud_w, 15 + hud_h), (0, 210, 255), 2)

        cv2.putText(frame, f"MODEL:   {model_name[:24]}", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(frame, f"TRACKER: {tracker_name.upper()}", (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
        cv2.putText(frame, f"LIVE FPS:{fps:5.1f} | Inf: {inference_ms:4.1f}ms", (28, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1)
        cv2.putText(frame, f"NMS-Free:{postprocess_ms:4.2f}ms | Active: {active_count:02d}", (28, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 50), 1)
        cv2.putText(frame, f"TOTAL UNIQUE IDS: {unique_total}", (28, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 255), 1)
        return frame

    def draw_tracks(
        self,
        frame: np.ndarray,
        boxes_result: Any
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Vẽ bounding box, ID tag và vệt quỹ đạo di chuyển (Trajectory Trail).
        
        Returns:
            Tuple gồm khung hình đã vẽ và danh sách ID đang kích hoạt (active_ids).
        """
        active_ids: List[int] = []
        if boxes_result is None or boxes_result.id is None:
            return frame, active_ids

        track_ids = boxes_result.id.int().cpu().tolist()
        xyxys = boxes_result.xyxy.cpu().numpy()
        confs = boxes_result.conf.cpu().numpy() if boxes_result.conf is not None else [0.0]*len(track_ids)

        for xyxy, track_id, conf in zip(xyxys, track_ids, confs):
            active_ids.append(track_id)
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Cập nhật vệt quỹ đạo
            self.track_history.setdefault(track_id, []).append((cx, cy))
            if len(self.track_history[track_id]) > self.max_trail_len:
                self.track_history[track_id].pop(0)

            color = self.get_color(track_id)

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Vẽ nhãn ID & Conf
            label = f"#{track_id} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

            # Vẽ vệt quỹ đạo (Trajectory Line) nếu show_trail được bật
            if self.show_trail:
                pts = np.array(self.track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

        return frame, active_ids
