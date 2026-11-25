"""
Module trực quan hóa kết quả bám vết (Bounding box, Trajectory trail, UAV Telemetry HUD).
Hỗ trợ hiển thị nhãn mục tiêu khóa [TARGET LOCK] và mục tiêu tái nhận diện [RE-ID].
"""

from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np


class TrackingVisualizer:
    """
    Lớp trực quan hóa chuyên dụng cho bám vết đối tượng người từ UAV.
    """

    def __init__(self, max_trail_len: int = 35, show_trail: bool = False):
        self.max_trail_len = max_trail_len
        self.show_trail = show_trail
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}

    def reset(self):
        """Xóa lịch sử vệt quỹ đạo cho video mới."""
        self.track_history.clear()

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
        unique_total: int,
        reacquired_total: int = 0,
        locked_id: Optional[int] = None
    ) -> np.ndarray:
        """Vẽ bảng đo lường Telemetry HUD nhỏ gọn, thanh thoát góc trên bên trái."""
        hud_w, hud_h = 220, 84
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), (0, 180, 220), 1)

        # Dòng 1: Mô hình & Thuật toán
        cv2.putText(frame, f"{model_name[:14]} | {tracker_name.upper()}", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        # Dòng 2: Tốc độ xử lý & Độ trễ
        cv2.putText(frame, f"FPS: {fps:4.1f} | Inf: {inference_ms:4.1f}ms", (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 120), 1)
        # Dòng 3: Số mục tiêu active & Tổng ID
        cv2.putText(frame, f"Active: {active_count:02d} | Total IDs: {unique_total}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 215, 0), 1)
        # Dòng 4: Tái nhận diện Re-ID
        mem_text = f"Re-ID: {reacquired_total} recovered"
        if locked_id is not None:
            mem_text += f" | LOCK: #{locked_id}"
        cv2.putText(frame, mem_text, (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 240, 220), 1)

        return frame

    def draw_tracks(
        self,
        frame: np.ndarray,
        boxes_result: Any,
        reacquired_ids: Optional[List[int]] = None,
        locked_id: Optional[int] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Vẽ bounding box, ID tag và vệt quỹ đạo di chuyển (Trajectory Trail).
        
        Returns:
            Tuple gồm khung hình đã vẽ và danh sách ID đang kích hoạt (active_ids).
        """
        active_ids: List[int] = []
        if boxes_result is None or boxes_result.id is None:
            return frame, active_ids

        reacquired_set = set(reacquired_ids or [])
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

            # Phân định màu và độ dày
            is_locked = (locked_id is not None and track_id == locked_id)
            is_reid = (track_id in reacquired_set)

            if is_locked:
                color = (0, 0, 255)  # Đỏ rực cho Locked Target
                thickness = 3
            elif is_reid:
                color = (0, 255, 200)  # Xanh ngọc cho Re-ID Target
                thickness = 2
            else:
                color = self.get_color(track_id)
                thickness = 2

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Vẽ nhãn ID & Conf nhỏ gọn vừa vặn
            label = f"#{track_id} ({conf:.2f})"
            if is_locked:
                label = f"LOCK #{track_id}"
            elif is_reid:
                label = f"#{track_id} [RE-ID]"

            font_scale = 0.36
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            text_color = (0, 0, 0) if is_reid else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

            # Vẽ vệt quỹ đạo (Trajectory Line) nếu show_trail được bật
            if self.show_trail:
                pts = np.array(self.track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

        return frame, active_ids
