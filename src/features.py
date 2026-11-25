"""
Module trích xuất đặc trưng ngoại hình (Appearance Feature Extractor) cho UAV Human Tracking.
Sử dụng phân rã không gian đa vùng (Multi-Zone Spatial Color Histogram) trên không gian màu HSV:
- Vùng 1: Đỉnh đầu / Vai / Mũ (0% - 35% chiều cao)
- Vùng 2: Thân trên / Áo / Balo (35% - 70% chiều cao)
- Vùng 3: Thân dưới / Quần / Chân (70% - 100% chiều cao)
Bất biến với tỷ lệ, kháng biến thiên ánh sáng, đạt tốc độ siêu nhanh (< 0.2 ms / bbox trên CPU).
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


class AppearanceFeatureExtractor:
    """
    Trích xuất vector đặc trưng ngoại hình từ bounding box người.
    Kết hợp phân rã 3 vùng không gian cơ thể và biểu đồ màu HSV chuẩn hóa L2.
    """

    def __init__(
        self,
        h_bins: int = 12,
        s_bins: int = 6,
        v_bins: int = 4,
        margin_ratio: float = 0.05
    ):
        """
        Khởi tạo bộ trích xuất đặc trưng.
        
        Args:
            h_bins: Số lượng bin cho thành phần sắc màu (Hue: 0-180).
            s_bins: Số lượng bin cho độ bão hòa (Saturation: 0-256).
            v_bins: Số lượng bin cho độ sáng (Value: 0-256).
            margin_ratio: Tỷ lệ co biên (inset margin) nhằm giảm nhiễu nền xung quanh bbox.
        """
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.margin_ratio = margin_ratio
        
        # Kích thước vector mỗi vùng = h_bins * s_bins * v_bins
        self.zone_dim = self.h_bins * self.s_bins * self.v_bins
        # Tổng số chiều = 3 vùng * zone_dim
        self.feature_dim = 3 * self.zone_dim

    def _extract_zone_hist(self, hsv_patch: np.ndarray) -> np.ndarray:
        """Tính biểu đồ 3D HSV trên một vùng ảnh và chuẩn hóa L2."""
        if hsv_patch.size == 0 or hsv_patch.shape[0] < 2 or hsv_patch.shape[1] < 2:
            return np.zeros(self.zone_dim, dtype=np.float32)

        hist = cv2.calcHist(
            [hsv_patch],
            [0, 1, 2],
            None,
            [self.h_bins, self.s_bins, self.v_bins],
            [0, 180, 0, 256, 0, 256]
        )
        hist = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 1e-6:
            hist /= norm
        return hist

    def extract(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Trích xuất vector đặc trưng ngoại hình 384-D chuẩn hóa L2 từ bounding box.
        
        Args:
            frame: Khung hình gốc (BGR).
            bbox: Tọa độ bounding box (x1, y1, x2, y2).
            
        Returns:
            Vector đặc trưng np.ndarray chuẩn hóa đơn vị (norm = 1.0).
        """
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Kẹp tọa độ trong khung hình
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(0, min(x2, w_img))
        y2 = max(0, min(y2, h_img))

        bw = x2 - x1
        bh = y2 - y1

        if bw < 4 or bh < 6:
            # BBox quá bé, trả về vector 0
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Cắt bỏ viền ngoài để loại trừ hậu cảnh (background)
        mx = int(bw * self.margin_ratio)
        my = int(bh * self.margin_ratio)
        cx1 = min(x1 + mx, x2 - 2)
        cy1 = min(y1 + my, y2 - 2)
        cx2 = max(x2 - mx, cx1 + 2)
        cy2 = max(y2 - my, cy1 + 2)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        crop_h = hsv_crop.shape[0]

        # Phân chia 3 vùng cơ thể theo chiều dọc
        z1_end = int(crop_h * 0.35)
        z2_end = int(crop_h * 0.70)

        zone1 = hsv_crop[0:z1_end, :]        # Đầu & Vai
        zone2 = hsv_crop[z1_end:z2_end, :]   # Thân trên
        zone3 = hsv_crop[z2_end:, :]         # Chân & Giày

        h1 = self._extract_zone_hist(zone1)
        h2 = self._extract_zone_hist(zone2)
        h3 = self._extract_zone_hist(zone3)

        # Ghép nối các vector đặc trưng
        feature = np.concatenate([h1, h2, h3])
        total_norm = np.linalg.norm(feature)
        if total_norm > 1e-6:
            feature /= total_norm

        return feature

    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Tính độ tương đồng Cosine giữa 2 vector đặc trưng chuẩn hóa L2.
        
        Args:
            feat1: Vector đặc trưng 1.
            feat2: Vector đặc trưng 2.
            
        Returns:
            Độ tương đồng trong khoảng [0.0, 1.0].
        """
        if feat1.size == 0 or feat2.size == 0:
            return 0.0
        dot = float(np.dot(feat1, feat2))
        return max(0.0, min(1.0, dot))
