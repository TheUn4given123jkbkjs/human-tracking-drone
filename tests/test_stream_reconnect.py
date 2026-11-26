"""
Unit test kiểm thử tính năng chịu lỗi mất sóng Wi-Fi (Auto-Reconnect & Frame Hold).
"""

import sys
import os
import time
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.stream_receiver import BaseStreamReceiver, StreamReceiverFactory
from src.detector import UAVDetector
from src.tracker import UAVTracker
from src.visualizer import TrackingVisualizer
from src.pipeline import UAVTrackingPipeline

@StreamReceiverFactory.register("mock_wifi_test")
class MockWifiDroneReceiver(BaseStreamReceiver):
    """Giả lập luồng drone Wi-Fi: phát 10 frames, mất sóng 1.5s, rồi phát tiếp 15 frames."""
    def __init__(self, source, **kwargs):
        super().__init__(source)
        self.source = str(ROOT / source) if not Path(source).is_absolute() else str(source)
        self.is_live = True
        self.cap = None
        self.total_read = 0
        self.drop_start = None
        self.drop_simulated = False
        self._running = False

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = 30.0
        self._running = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._running:
            return False, None
        
        # Sau 10 frame, mô phỏng rớt mạng Wi-Fi trong 1.5 giây
        if self.total_read == 10 and not self.drop_simulated:
            if self.drop_start is None:
                self.drop_start = time.perf_counter()
            
            elapsed = time.perf_counter() - self.drop_start
            if elapsed < 1.5:
                time.sleep(0.05)
                return False, None
            else:
                self.drop_simulated = True

        ret, frame = self.cap.read()
        if not ret or self.total_read >= 25:
            self._running = False
            return False, None

        self.total_read += 1
        return True, frame

    def is_opened(self) -> bool:
        return self._running

    def release(self):
        self._running = False
        if self.cap is not None:
            self.cap.release()

def test_live_reconnect():
    detector = UAVDetector("yolo26n_merged")
    tracker = UAVTracker(detector=detector, tracker_type="bytetrack", use_memory=True)
    visualizer = TrackingVisualizer()
    pipeline = UAVTrackingPipeline(detector, tracker, visualizer)

    video_source = "data/gettyimages-1345327468-640_adpp.mp4"
    receiver = MockWifiDroneReceiver(video_source)

    out_path = ROOT / "runs/test_wifi_reconnect.mp4"
    metrics = pipeline.run_video(
        source_path=receiver,
        output_path=str(out_path),
        max_frames=25
    )

    assert metrics.total_frames == 25, f"Kỳ vọng 25 frames, thực tế: {metrics.total_frames}"
    assert receiver.drop_simulated, "Kịch bản mất sóng chưa được kích hoạt!"
    if out_path.exists():
        out_path.unlink()
    print("  --> PASS: test_live_reconnect thành công.")

if __name__ == "__main__":
    test_live_reconnect()
