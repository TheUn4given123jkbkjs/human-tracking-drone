"""
Module Quản lý Bộ nhớ Ngoại hình (Appearance Memory Bank) cho UAV Human Tracking.
Chống phân mảnh quỹ đạo (ID switches), hàn gắn vết khi đối tượng bị che khuất tạm thời
hoặc camera drone chao liệng xoay góc bằng cách tái nhận diện (Re-ID) dựa trên độ tương đồng Cosine.
"""

from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import numpy as np

from src.features import AppearanceFeatureExtractor


class TrackMemory:
    """
    Lưu trữ thông tin lịch sử không gian - thời gian và đặc trưng ngoại hình của một đối tượng.
    """

    def __init__(
        self,
        track_id: int,
        initial_feature: np.ndarray,
        initial_bbox: Tuple[int, int, int, int],
        frame_idx: int,
        max_history: int = 15,
        ema_alpha: float = 0.25
    ):
        self.track_id = track_id
        self.max_history = max_history
        self.ema_alpha = ema_alpha

        # Hàng đợi lưu trữ các vector đặc trưng gần nhất
        self.feature_history: deque = deque(maxlen=max_history)
        self.feature_history.append(initial_feature)
        self.mean_feature: np.ndarray = initial_feature.copy()

        self.last_bbox: Tuple[int, int, int, int] = initial_bbox
        self.last_center: Tuple[float, float] = self._calc_center(initial_bbox)
        self.last_seen_frame: int = frame_idx
        self.first_seen_frame: int = frame_idx

        # Observation-Centric Momentum (OCM): Lịch sử tâm và vector vận tốc tức thời
        self.center_history: deque = deque(maxlen=max_history)
        self.center_history.append((self.last_center, frame_idx))
        self.velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) pixel / frame
        self.speed: float = 0.0
        self.heading_angle: float = 0.0  # radian [-pi, pi]

        self.hits: int = 1
        self.lost_frames: int = 0
        self.age: int = 1
        self.is_locked: bool = False

    @staticmethod
    def _calc_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(
        self,
        feature: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_idx: int
    ) -> None:
        """Cập nhật bộ nhớ với quan sát thực tế mới (Observation-Centric Update)."""
        self.feature_history.append(feature)

        # Cập nhật trung bình động EMA của vector đặc trưng
        if self.mean_feature.size > 0 and feature.size > 0:
            self.mean_feature = (1.0 - self.ema_alpha) * self.mean_feature + self.ema_alpha * feature
            norm = np.linalg.norm(self.mean_feature)
            if norm > 1e-6:
                self.mean_feature /= norm

        new_center = self._calc_center(bbox)
        dt = max(1, frame_idx - self.last_seen_frame)

        # Tính toán vector vận tốc tức thời thực tế từ quan sát (OCM)
        vx = (new_center[0] - self.last_center[0]) / float(dt)
        vy = (new_center[1] - self.last_center[1]) / float(dt)

        # Làm mượt vận tốc bằng EMA để khử nhiễu jitter của BBox
        if self.hits > 1:
            self.velocity = (0.7 * self.velocity[0] + 0.3 * vx, 0.7 * self.velocity[1] + 0.3 * vy)
        else:
            self.velocity = (vx, vy)

        self.speed = float(np.hypot(self.velocity[0], self.velocity[1]))
        if self.speed > 0.5:
            self.heading_angle = float(np.arctan2(self.velocity[1], self.velocity[0]))

        self.last_bbox = bbox
        self.last_center = new_center
        self.center_history.append((new_center, frame_idx))
        self.last_seen_frame = frame_idx
        self.lost_frames = 0
        self.hits += 1
        self.age = frame_idx - self.first_seen_frame + 1

    def predict_center(self, target_frame_idx: int) -> Tuple[float, float]:
        """Dự phóng vị trí tâm đối tượng theo vector quán tính thực tế (OCM Extrapolation)."""
        dt = max(0, target_frame_idx - self.last_seen_frame)
        pred_x = self.last_center[0] + self.velocity[0] * dt
        pred_y = self.last_center[1] + self.velocity[1] * dt
        return (pred_x, pred_y)

    def compute_similarity(self, candidate_feature: np.ndarray) -> float:
        """
        Tính điểm tương đồng kết hợp giữa EMA mean feature và bộ nhớ lịch sử.
        
        Returns:
            Điểm tương đồng Cosine trong khoảng [0.0, 1.0].
        """
        if candidate_feature.size == 0 or self.mean_feature.size == 0:
            return 0.0

        # Tương đồng với Mean Feature
        sim_mean = AppearanceFeatureExtractor.cosine_similarity(candidate_feature, self.mean_feature)

        # Tương đồng lớn nhất với các template trong lịch sử
        sim_max = 0.0
        for hist_feat in self.feature_history:
            s = AppearanceFeatureExtractor.cosine_similarity(candidate_feature, hist_feat)
            if s > sim_max:
                sim_max = s

        # Trọng số kết hợp: 60% trung bình cộng EMA + 40% đỉnh lịch sử
        return 0.60 * sim_mean + 0.40 * sim_max


class AppearanceMemoryBank:
    """
    Hệ thống ngân hàng bộ nhớ ngoại hình đa đối tượng.
    Tự động liên kết các track mới sinh với các track bị mất dấu tạm thời.
    """

    def __init__(
        self,
        feature_extractor: Optional[AppearanceFeatureExtractor] = None,
        similarity_thresh: float = 0.62,
        max_lost_frames: int = 60,
        max_spatial_distance: float = 450.0,
        max_history: int = 15,
        mode: str = "all"
    ):
        """
        Khởi tạo Memory Bank.
        
        Args:
            feature_extractor: Bộ trích xuất đặc trưng ngoại hình HSV.
            similarity_thresh: Ngưỡng Cosine Similarity để chấp nhận tái nhận diện (Re-ID).
            max_lost_frames: Số frame tối đa giữ một track trong bộ nhớ trước khi giải phóng.
            max_spatial_distance: Bán kính pixel tối đa để so khớp không gian (chống nhầm người ở hai đầu video).
            max_history: Số lượng template lưu trữ cho mỗi track.
            mode: 'all' (lưu vết toàn bộ ID) hoặc 'locked_only' (chỉ tối ưu cho mục tiêu chỉ định).
        """
        self.extractor = feature_extractor or AppearanceFeatureExtractor()
        self.similarity_thresh = similarity_thresh
        self.max_lost_frames = max_lost_frames
        self.max_spatial_distance = max_spatial_distance
        self.max_history = max_history
        self.mode = mode.lower()

        # Bảng ánh xạ: raw_id từ base tracker -> canonical_id của memory bank
        self.id_remap_table: Dict[int, int] = {}
        self.next_canonical_id: int = 1

        # Danh bạ các track đang hoạt động và các track bị mất dấu
        self.active_tracks: Dict[int, TrackMemory] = {}
        self.lost_tracks: Dict[int, TrackMemory] = {}

        # Mục tiêu được chỉ định khóa (Lock Target)
        self.locked_target_id: Optional[int] = None

        # Thống kê hiệu năng
        self.reacquired_events: List[Dict[str, Any]] = []
        self.total_reacquired_count: int = 0

    def lock_target(self, target_id: int) -> bool:
        """Khóa một ID cụ thể làm mục tiêu chính."""
        canonical_id = self.id_remap_table.get(target_id, target_id)
        if canonical_id in self.active_tracks:
            self.locked_target_id = canonical_id
            self.active_tracks[canonical_id].is_locked = True
            print(f"[*] [MemoryBank] ĐÃ KHÓA MỤC TIÊU CHÍNH: ID #{canonical_id}")
            return True
        elif canonical_id in self.lost_tracks:
            self.locked_target_id = canonical_id
            self.lost_tracks[canonical_id].is_locked = True
            print(f"[*] [MemoryBank] ĐÃ KHÓA MỤC TIÊU (Hiện đang trong bộ nhớ Lost): ID #{canonical_id}")
            return True
        print(f"[!] [MemoryBank] Không tìm thấy ID #{target_id} để khóa mục tiêu.")
        return False

    def unlock_target(self) -> None:
        """Hủy khóa mục tiêu chính."""
        if self.locked_target_id is not None:
            if self.locked_target_id in self.active_tracks:
                self.active_tracks[self.locked_target_id].is_locked = False
            if self.locked_target_id in self.lost_tracks:
                self.lost_tracks[self.locked_target_id].is_locked = False
            print(f"[*] [MemoryBank] Đã hủy khóa mục tiêu ID #{self.locked_target_id}")
            self.locked_target_id = None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        raw_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Quy trình xử lý bộ nhớ ngoại hình chính cho từng frame:
        1. Ánh xạ raw IDs qua bảng re-map đã biết.
        2. Trích xuất đặc trưng ngoại hình cho từng detection.
        3. Với các ID mới xuất hiện, truy vấn ngân hàng Lost Tracks để Re-ID.
        4. Cập nhật bộ nhớ active, luân chuyển các track không được phát hiện sang lost.
        
        Args:
            frame: Khung hình video gốc (BGR).
            frame_idx: Số thứ tự frame hiện tại.
            raw_detections: Danh sách các detection từ Base Tracker:
                [{'raw_id': int, 'bbox': (x1, y1, x2, y2), 'conf': float}]
                
        Returns:
            Danh sách detection đã được chuẩn hóa ID qua Memory Bank:
                [{'track_id': int, 'raw_id': int, 'bbox': ..., 'conf': ..., 'reacquired': bool, 'sim': float}]
        """
        results: List[Dict[str, Any]] = []
        current_frame_canonical_ids = set()

        # Bước 1: Trích xuất đặc trưng và chuẩn bị danh sách candidate
        candidates = []
        for det in raw_detections:
            raw_id = det["raw_id"]
            bbox = det["bbox"]
            conf = det["conf"]

            feat = self.extractor.extract(frame, bbox)
            canonical_id = self.id_remap_table.get(raw_id, raw_id)
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

            candidates.append({
                "raw_id": raw_id,
                "canonical_id": canonical_id,
                "bbox": bbox,
                "conf": conf,
                "feat": feat,
                "center": center,
                "reacquired": False,
                "sim": 0.0
            })

        # Bước 2: Phân loại tiếp tục bám vết hoặc Re-ID
        for cand in candidates:
            cid = cand["canonical_id"]
            raw_id = cand["raw_id"]
            bbox = cand["bbox"]
            feat = cand["feat"]
            center = cand["center"]

            # Trường hợp 1: ID đã nằm trong active tracks
            if cid in self.active_tracks:
                self.active_tracks[cid].update(feat, bbox, frame_idx)
                current_frame_canonical_ids.add(cid)
                cand["track_id"] = cid
                results.append(cand)
                continue

            # Trường hợp 2: ID là một ID mới đối với active tracks
            # Kiểm tra xem đây có phải là một ID trong lost_tracks quay lại không
            matched_lost_id = None
            best_sim = 0.0

            # Ưu tiên kiểm tra: nếu chính ID này từng bị lost gần đây
            if cid in self.lost_tracks:
                lost_track = self.lost_tracks[cid]
                s = lost_track.compute_similarity(feat)
                pred_c = lost_track.predict_center(frame_idx)
                dist = min(
                    float(np.hypot(center[0] - lost_track.last_center[0], center[1] - lost_track.last_center[1])),
                    float(np.hypot(center[0] - pred_c[0], center[1] - pred_c[1]))
                )
                if s >= (self.similarity_thresh - 0.08) and dist <= self.max_spatial_distance:
                    matched_lost_id = cid
                    best_sim = s

            # Nếu chưa khớp hoặc là một raw_id hoàn toàn mới, quét toàn bộ lost_tracks
            if matched_lost_id is None and len(self.lost_tracks) > 0:
                for lost_id, lost_track in list(self.lost_tracks.items()):
                    # Nếu đang ở chế độ locked_only, chỉ so khớp với locked_target_id
                    if self.mode == "locked_only" and self.locked_target_id is not None:
                        if lost_id != self.locked_target_id:
                            continue

                    sim = lost_track.compute_similarity(feat)
                    pred_c = lost_track.predict_center(frame_idx)
                    dist = min(
                        float(np.hypot(center[0] - lost_track.last_center[0], center[1] - lost_track.last_center[1])),
                        float(np.hypot(center[0] - pred_c[0], center[1] - pred_c[1]))
                    )

                    # Điều kiện tái nhận diện: Độ tương đồng ngoại hình cao và khoảng cách không gian (hoặc tâm dự phóng OCM) hợp lý
                    if sim >= self.similarity_thresh and dist <= self.max_spatial_distance:
                        if sim > best_sim:
                            best_sim = sim
                            matched_lost_id = lost_id

            # Xử lý kết quả so khớp
            if matched_lost_id is not None:
                # TÁI NHẬN DIỆN THÀNH CÔNG (RE-IDENTIFICATION SUCCESS)
                reacquired_track = self.lost_tracks.pop(matched_lost_id)
                reacquired_track.update(feat, bbox, frame_idx)
                self.active_tracks[matched_lost_id] = reacquired_track

                # Lưu vào bảng ánh xạ ID để các frame tiếp theo Base Tracker trả về raw_id vẫn hiểu là canonical_id
                self.id_remap_table[raw_id] = matched_lost_id
                current_frame_canonical_ids.add(matched_lost_id)

                self.total_reacquired_count += 1
                self.reacquired_events.append({
                    "frame": frame_idx,
                    "raw_id": raw_id,
                    "canonical_id": matched_lost_id,
                    "sim": best_sim
                })

                cand["track_id"] = matched_lost_id
                cand["reacquired"] = True
                cand["sim"] = best_sim
                results.append(cand)
            else:
                # ĐỐI TƯỢNG MỚI HOÀN TOÀN
                if raw_id in self.id_remap_table:
                    assigned_id = self.id_remap_table[raw_id]
                else:
                    assigned_id = self.next_canonical_id
                    self.next_canonical_id += 1
                    self.id_remap_table[raw_id] = assigned_id

                new_track = TrackMemory(
                    track_id=assigned_id,
                    initial_feature=feat,
                    initial_bbox=bbox,
                    frame_idx=frame_idx,
                    max_history=self.max_history
                )
                self.active_tracks[assigned_id] = new_track
                current_frame_canonical_ids.add(assigned_id)
                cand["track_id"] = assigned_id
                results.append(cand)

        # Bước 3: Luân chuyển các track không quan sát được sang lost_tracks
        active_ids_snapshot = list(self.active_tracks.keys())
        for aid in active_ids_snapshot:
            if aid not in current_frame_canonical_ids:
                track = self.active_tracks.pop(aid)
                track.lost_frames += 1
                self.lost_tracks[aid] = track

        # Bước 4: Tăng lost_frames cho các track đã ở trong lost_tracks và dọn dẹp các track hết hạn
        lost_ids_snapshot = list(self.lost_tracks.keys())
        for lid in lost_ids_snapshot:
            if lid not in current_frame_canonical_ids:
                # Nếu đã được xử lý ở bước 3 thì không tăng 2 lần
                if lid not in active_ids_snapshot:
                    self.lost_tracks[lid].lost_frames += 1

                # Nếu vượt quá thời gian lưu vết tối đa (ngoại trừ target đang được khóa)
                max_allowed = self.max_lost_frames * 3 if self.lost_tracks[lid].is_locked else self.max_lost_frames
                if self.lost_tracks[lid].lost_frames > max_allowed:
                    del self.lost_tracks[lid]

        return results

    def reset(self):
        """Khởi động lại toàn bộ ngân hàng bộ nhớ cho video mới (bắt đầu ID từ 1)."""
        self.next_canonical_id = 1
        self.id_remap_table.clear()
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.reacquired_events.clear()
        self.total_reacquired_count = 0
        self.locked_target_id = None

    def get_summary(self) -> Dict[str, Any]:
        """Tổng hợp thống kê hoạt động của bộ nhớ."""
        return {
            "total_reacquired": self.total_reacquired_count,
            "active_tracks_count": len(self.active_tracks),
            "lost_tracks_count": len(self.lost_tracks),
            "unique_canonical_ids": len(self.active_tracks) + len(self.lost_tracks) + len(self.id_remap_table),
            "events": self.reacquired_events
        }
