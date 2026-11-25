"""
Module trích xuất đặc trưng ngoại hình (Appearance Feature Extractor) cho UAV Human Tracking.
NÂNG CẤP TỔNG QUÁT: Tích hợp cơ chế Phân rã Đường tròn/Elip Đồng tâm (Concentric Radial Slicing)
kết hợp Bộ điều phối Thích ứng theo Tỷ lệ Khung hình (Aspect-Ratio Aware Adaptive Slicing).

Bao phủ 100% các dải góc nhìn camera của UAV:
1. Góc nghiêng (Oblique 15° - 50°, h/w >= 1.65):
   - Phân rã 3 dải dọc: Đỉnh đầu/Vai (0-35%), Thân trên/Áo (35-70%), Chân/Quần (70-100%).
2. Góc chúc dốc & Đỉnh đầu thẳng đứng (Nadir / Bird's-Eye View 50° - 90°, h/w < 1.65):
   - Phân rã 3 vành elip đồng tâm (Concentric Rings):
     • Vành 1 - Tâm lõi (r <= 0.40 R): Đỉnh đầu, tóc, mũ.
     • Vành 2 - Vòng giữa (0.40 R < r <= 0.75 R): Vai áo, lưng, ngực, balo.
     • Vành 3 - Vành ngoài (0.75 R < r <= 1.10 R): Cánh tay, hông, chân.
   - BẤT BIẾN HOÀN TOÀN VỚI HƯỚNG XOAY NGƯỜI (Rotation-Invariant) dưới góc nhìn thẳng đứng!
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


class AppearanceFeatureExtractor:
    """
    Trích xuất vector đặc trưng ngoại hình bất biến góc nhìn (Viewpoint-Invariant) từ bounding box.
    Hỗ trợ chế độ Thích ứng tự động (Adaptive), Phân rã dọc (Vertical) và Đồng tâm (Radial).
    """

    def __init__(
        self,
        h_bins: int = 12,
        s_bins: int = 6,
        v_bins: int = 4,
        margin_ratio: float = 0.05,
        aspect_ratio_thresh: float = 1.65,
        slicing_mode: str = "adaptive"
    ):
        """
        Khởi tạo bộ trích xuất đặc trưng ngoại hình.
        
        Args:
            h_bins: Số bin cho Hue (0-180).
            s_bins: Số bin cho Saturation (0-256).
            v_bins: Số bin cho Value (0-256).
            margin_ratio: Tỷ lệ co viền loại trừ nhiễu nền.
            aspect_ratio_thresh: Ngưỡng tỷ lệ h/w để chuyển đổi giữa Oblique và Nadir.
            slicing_mode: 'adaptive' (tự động theo góc), 'radial' (luôn đồng tâm), 'vertical' (luôn cắt dọc).
        """
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.margin_ratio = margin_ratio
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.slicing_mode = slicing_mode.lower()

        # Mỗi vùng: h_bins * s_bins * v_bins (12 * 6 * 4 = 288 chiều)
        self.zone_dim = self.h_bins * self.s_bins * self.v_bins
        # 3 vùng không gian (Đầu, Thân, Chân/Vành) = 3 * 288 = 864 chiều
        self.feature_dim = 3 * self.zone_dim

    def _extract_zone_hist(self, hsv_img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Tính biểu đồ 3D HSV trên một vùng có mask chỉ định và chuẩn hóa L2."""
        if hsv_img.size == 0 or hsv_img.shape[0] < 2 or hsv_img.shape[1] < 2:
            return np.zeros(self.zone_dim, dtype=np.float32)

        if mask is not None and np.count_nonzero(mask) < 4:
            return np.zeros(self.zone_dim, dtype=np.float32)

        hist = cv2.calcHist(
            [hsv_img],
            [0, 1, 2],
            mask,
            [self.h_bins, self.s_bins, self.v_bins],
            [0, 180, 0, 256, 0, 256]
        )
        hist = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 1e-6:
            hist /= norm
        return hist

    def _extract_vertical_features(self, hsv_crop: np.ndarray) -> np.ndarray:
        """Phân rã 3 dải ngang dọc cho góc nghiêng (Oblique 15° - 50°)."""
        crop_h = hsv_crop.shape[0]
        z1_end = int(crop_h * 0.35)
        z2_end = int(crop_h * 0.70)

        zone1 = hsv_crop[0:z1_end, :]        # Đầu & Vai
        zone2 = hsv_crop[z1_end:z2_end, :]   # Thân trên / Áo
        zone3 = hsv_crop[z2_end:, :]         # Thân dưới / Quần

        h1 = self._extract_zone_hist(zone1)
        h2 = self._extract_zone_hist(zone2)
        h3 = self._extract_zone_hist(zone3)

        return np.concatenate([h1, h2, h3])

    def _extract_radial_features(self, hsv_crop: np.ndarray) -> np.ndarray:
        """
        Phân rã 3 vành elip đồng tâm (Concentric Radial Slicing) cho góc chúc dốc (Nadir 50° - 90°).
        Bất biến với hướng quay người 360 độ (Rotation-Invariant).
        """
        crop_h, crop_w = hsv_crop.shape[:2]
        y, x = np.ogrid[:crop_h, :crop_w]
        cy, cx = crop_h / 2.0, crop_w / 2.0
        ry, rx = max(1.0, cy), max(1.0, cx)

        # Khoảng cách elip chuẩn hóa bình phương từ tâm
        dist_sq = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2

        # Vành 1: Lõi tâm (r <= 0.40) -> Đỉnh đầu, tóc, nón mũ
        m1 = (dist_sq <= (0.40 ** 2)).astype(np.uint8) * 255
        # Vành 2: Vòng giữa (0.40 < r <= 0.75) -> Vai áo, ngực, lưng, balo
        m2 = ((dist_sq > (0.40 ** 2)) & (dist_sq <= (0.75 ** 2))).astype(np.uint8) * 255
        # Vành 3: Vòng ngoài (0.75 < r <= 1.15) -> Cánh tay, hông, chân
        m3 = ((dist_sq > (0.75 ** 2)) & (dist_sq <= (1.15 ** 2))).astype(np.uint8) * 255

        h1 = self._extract_zone_hist(hsv_crop, mask=m1)
        h2 = self._extract_zone_hist(hsv_crop, mask=m2)
        h3 = self._extract_zone_hist(hsv_crop, mask=m3)

        return np.concatenate([h1, h2, h3])

    def extract(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Trích xuất vector đặc trưng ngoại hình 864-D chuẩn hóa L2 từ bounding box.
        Tự động nhận diện góc nhìn thông qua tỷ lệ hình học h/w của BBox.
        
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

        if bw < 4 or bh < 4:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Cắt bỏ viền ngoài để giảm thiểu nhiễu nền
        mx = int(bw * self.margin_ratio)
        my = int(bh * self.margin_ratio)
        cx1 = min(x1 + mx, x2 - 2)
        cy1 = min(y1 + my, y2 - 2)
        cx2 = max(x2 - mx, cx1 + 2)
        cy2 = max(y2 - my, cy1 + 2)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return np.zeros(self.feature_dim, dtype=np.float32)

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        aspect_ratio = float(bh) / max(1.0, float(bw))

        # Điều phối chế độ trích xuất thích ứng theo góc nhìn camera
        if self.slicing_mode == "radial":
            feature = self._extract_radial_features(hsv_crop)
        elif self.slicing_mode == "vertical":
            feature = self._extract_vertical_features(hsv_crop)
        else:
            # Chế độ tự động 'adaptive':
            # Nếu BBox đứng cao (h/w >= 1.65): Góc nghiêng (Oblique) -> Cắt dải dọc
            # Nếu BBox gần vuông (h/w < 1.65): Góc dốc (Nadir) -> Cắt vòng tròn đồng tâm
            if aspect_ratio >= self.aspect_ratio_thresh:
                feature = self._extract_vertical_features(hsv_crop)
            else:
                feature = self._extract_radial_features(hsv_crop)

        # Chuẩn hóa L2 norm toàn phần
        total_norm = np.linalg.norm(feature)
        if total_norm > 1e-6:
            feature /= total_norm

        return feature

    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Tính độ tương đồng Cosine giữa 2 vector đặc trưng chuẩn hóa L2.
        
        Returns:
            Độ tương đồng trong khoảng [0.0, 1.0].
        """
        if feat1.size == 0 or feat2.size == 0:
            return 0.0
        dot = float(np.dot(feat1, feat2))
        return max(0.0, min(1.0, dot))
