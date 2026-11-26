"""
Module Thu nhận Luồng Video Đa Nguồn (Generic Stream Receiver).

  - CameraStreamReceiver ("camera"): Webcam laptop / USB UVC camera.
  - RTSPStreamReceiver   ("rtsp"):   Luồng Wi-Fi từ drone (rtsp://, udp://, http://),
                                     worker thread chỉ giữ frame mới nhất (chống trôi độ trễ).
  - FileStreamReceiver   ("file"):   Tệp video cục bộ.

Thêm nguồn mới: kế thừa BaseStreamReceiver và gắn @StreamReceiverFactory.register("tên").
Pipeline không cần sửa.
"""

import os
import re
import time
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Type
import numpy as np
import cv2


class BaseStreamReceiver(ABC):
    """Interface chung cho mọi nguồn video."""

    def __init__(self, source: Any):
        self.source = source
        self.name: str = str(source)      # Dùng để đặt tên file kết quả
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0
        self.total_frames: int = 0        # 0 với luồng live
        self.is_live: bool = False

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]: ...

    @abstractmethod
    def is_opened(self) -> bool: ...

    @abstractmethod
    def release(self): ...


class StreamReceiverFactory:
    """Registry khởi tạo receiver theo cờ stream_type hoặc tự nhận diện từ source."""
    _REGISTRY: Dict[str, Type[BaseStreamReceiver]] = {}

    @classmethod
    def register(cls, type_name: str):
        def decorator(subclass: Type[BaseStreamReceiver]):
            cls._REGISTRY[type_name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, source: Any, stream_type: Optional[str] = None, **kwargs) -> BaseStreamReceiver:
        st = (stream_type or "auto").lower()
        if st == "auto":
            src = str(source).strip().lower()
            if src.startswith(("rtsp://", "rtsps://", "udp://", "http://", "https://")):
                st = "rtsp"
            elif src.isdigit():
                st = "camera"
            else:
                st = "file"
        if st not in cls._REGISTRY:
            raise ValueError(f"Loại luồng '{stream_type}' không tồn tại. Hỗ trợ: {', '.join(cls._REGISTRY)}")
        return cls._REGISTRY[st](source, **kwargs)


@StreamReceiverFactory.register("camera")
class CameraStreamReceiver(BaseStreamReceiver):
    """Webcam laptop / USB UVC camera (DirectShow trên Windows)."""

    def __init__(self, source: Any = 0, **kwargs):
        super().__init__(source)
        self.device_index = int(source)
        self.name = f"camera_{self.device_index}"
        self.is_live = True
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device_index)
        ok, _ = self.cap.read() if self.cap.isOpened() else (False, None)
        if not ok:
            print(f"[!] [CameraReceiver] Không mở được camera #{self.device_index}.")
            return False
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


@StreamReceiverFactory.register("rtsp")
class RTSPStreamReceiver(BaseStreamReceiver):
    """
    Luồng video Wi-Fi từ drone (RTSP/UDP/HTTP).
    Worker thread đọc liên tục và chỉ giữ frame mới nhất (chống trôi độ trễ).
    Tích hợp cơ chế Auto-Reconnect tự động khôi phục kết nối khi rớt sóng Wi-Fi/mất gói,
    không ngắt luồng nhận diện của trạm mặt đất GCS.
    """

    def __init__(self, source: str, transport: str = "udp", timeout_sec: float = 0.1, **kwargs):
        super().__init__(source)
        self.url = str(source)
        self.name = "live_" + re.sub(r"[^A-Za-z0-9]+", "_", self.url).strip("_")
        self.transport = transport.lower()
        self.timeout_sec = timeout_sec
        self.is_live = True
        self.cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cond = threading.Condition()
        self._latest: Optional[np.ndarray] = None
        self._seq = 0          # Số thứ tự frame worker đã nhận
        self._read_seq = 0     # Số thứ tự frame đã trả cho pipeline
        self.is_connected = False

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Tạo đối tượng VideoCapture tối ưu độ trễ thấp FFMPEG."""
        old_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        try:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.transport}|fflags;nobuffer|flags;low_delay|max_delay;500000"
            )
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        finally:
            if old_opts is not None:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = old_opts
            else:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        return cap if (cap is not None and cap.isOpened()) else None

    def open(self) -> bool:
        self.cap = self._open_capture()
        ok, frame = self.cap.read() if self.cap is not None else (False, None)
        if not ok:
            print(f"[!] [RTSPReceiver] Không nhận được frame khởi tạo từ: {self.url}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
        self.height, self.width = frame.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._latest, self._seq = frame, 1
        self._running = True
        self.is_connected = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return True

    def _reconnect(self) -> bool:
        """Thử kết nối lại luồng video khi mất sóng Wi-Fi."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        new_cap = self._open_capture()
        if new_cap is not None:
            ok, frame = new_cap.read()
            if ok and frame is not None:
                self.cap = new_cap
                self.is_connected = True
                with self._cond:
                    self._latest = frame
                    self._seq += 1
                    self._cond.notify_all()
                print(f"[✓] [RTSPReceiver] Đã tự động kết nối lại thành công tới: {self.url}")
                return True
            new_cap.release()
        return False

    def _worker(self):
        consecutive_fails = 0
        while self._running:
            if self.cap is None:
                self.is_connected = False
                time.sleep(1.0)
                if not self._running:
                    break
                self._reconnect()
                continue

            ok, frame = self.cap.read()
            if not ok:
                consecutive_fails += 1
                if consecutive_fails >= 25:
                    self.is_connected = False
                    self._reconnect()
                    consecutive_fails = 0
                else:
                    time.sleep(0.01)
                continue

            consecutive_fails = 0
            self.is_connected = True
            with self._cond:
                self._latest = frame
                self._seq += 1
                self._cond.notify()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Đọc frame mới nhất. Nếu chưa có frame mới hoặc mất sóng, trả về (False, None)
        nhưng KHÔNG ngắt luồng (để GCS thực hiện Frame Hold & Auto-Reconnect).
        """
        with self._cond:
            has_new = self._cond.wait_for(
                lambda: self._seq > self._read_seq or not self._running, timeout=self.timeout_sec
            )
            if not self._running:
                return False, None
            if not has_new:
                return False, None
            self._read_seq = self._seq
            return True, self._latest

    def is_opened(self) -> bool:
        return self._running

    def release(self):
        self._running = False
        self.is_connected = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


@StreamReceiverFactory.register("file")
class FileStreamReceiver(BaseStreamReceiver):
    """Tệp video cục bộ (.mp4, .avi, .mkv)."""

    def __init__(self, source: str, **kwargs):
        super().__init__(source)
        self.path = Path(source)
        self.name = self.path.stem
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if not self.path.exists():
            print(f"[!] [FileReceiver] Không tìm thấy video: {self.path}")
            return False
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            return False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
