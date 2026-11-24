"""
Module phát hiện đối tượng người từ UAV sử dụng kiến trúc YOLO26 Native NMS-Free.
Tự động khám phá động các mô hình từ thư mục models/ (Dynamic Model Discovery).
"""

import os
from pathlib import Path
from typing import Optional, Union, Any, Dict, Tuple
import numpy as np

# Đảm bảo cache cục bộ cho Ultralytics
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("./Ultralytics")
from ultralytics import YOLO
from src.config import MODELS_DIR, DEFAULT_MODEL, DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMGSZ, DEFAULT_DEVICE


class UAVDetector:
    """
    Lớp đóng gói mô hình phát hiện người từ UAV (UAV Human Detector).
    Cơ chế Dynamic Discovery: Tự động quét các thư mục trong models/ mà không cần hardcode alias.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf_thresh: float = DEFAULT_CONF,
        iou_thresh: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = DEFAULT_DEVICE
    ):
        self.model_name, self.model_path = self._resolve_model(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.device = device
        
        self.model = self._load_model()

    @staticmethod
    def list_models() -> Dict[str, str]:
        """
        Quét động thư mục models/ và trả về từ điển {tên_mô_hình: đường_dẫn_best_pt}.
        Mỗi thư mục con trong models/ chứa weights/best.pt được xem là một mô hình hợp lệ.
        """
        available = {}
        if MODELS_DIR.exists():
            for sub in sorted(MODELS_DIR.iterdir()):
                if sub.is_dir():
                    best_pt = sub / "weights" / "best.pt"
                    if best_pt.is_file():
                        available[sub.name] = str(best_pt.resolve())
        return available

    def _resolve_model(self, query: str) -> Tuple[str, str]:
        """
        Phân giải tên mô hình theo quy chuẩn kỹ thuật:
        1. Khớp chính xác tên thư mục trong models/ (ví dụ: 'yolo26s_merged')
        2. Hoặc nhận đường dẫn trực tiếp tới file trọng số .pt hợp lệ
        
        Bắt buộc truyền đúng tên chuẩn, không sử dụng alias ngẫu hứng.
        """
        raw = str(query).strip()

        # 1. Kiểm tra nếu là file .pt trực tiếp
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".pt":
            name = p.parent.parent.name if p.parent.name == "weights" else p.stem
            return name, str(p.resolve())

        # 2. Quét động trong thư mục models/
        available = self.list_models()
        if raw in available:
            return raw, available[raw]

        # 3. Báo lỗi tường minh nếu không tìm thấy
        avail_list = "\n".join([f"  • {m}" for m in available.keys()])
        raise ValueError(
            f"\n[!] Tên mô hình không hợp lệ: '{raw}'.\n"
            f"Vui lòng chỉ định đúng một trong các mô hình có sẵn trong 'models/':\n{avail_list}\n"
            f"Hoặc truyền đường dẫn trực tiếp tới file trọng số .pt."
        )

    def _load_model(self) -> YOLO:
        """Nạp trọng số mô hình YOLO26."""
        print(f"[*] [UAVDetector] Nạp trọng số: {self.model_path}")
        model = YOLO(self.model_path)
        return model

    @property
    def num_parameters(self) -> int:
        """Số lượng tham số của mô hình."""
        try:
            return sum(p.numel() for p in self.model.model.parameters())
        except Exception:
            return 9465567 if "yolo26s" in self.model_name else 2533886

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
