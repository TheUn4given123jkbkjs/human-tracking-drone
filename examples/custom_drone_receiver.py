"""
Ví dụ: Thêm bộ thu video mới cho drone mà không sửa UAVTrackingPipeline.

  1. Kế thừa BaseStreamReceiver, cài đặt open/read/is_opened/release.
  2. Gắn @StreamReceiverFactory.register("tên").
  3. Đảm bảo module được import trước khi gọi pipeline, rồi dùng stream_type="tên"
     (hoặc truyền thẳng instance vào run_video).

Chạy thử: python examples/custom_drone_receiver.py <file_video_hoặc_url>
"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stream_receiver import BaseStreamReceiver, StreamReceiverFactory


@StreamReceiverFactory.register("my_drone")
class MyDroneReceiver(BaseStreamReceiver):
    def __init__(self, source, **kwargs):
        super().__init__(source)
        self.name = "my_drone"
        self.is_live = True
        self.cap = None

    def open(self) -> bool:
        # Thay bằng lệnh SDK của drone (vd: gửi 'streamon') rồi mở luồng
        self.cap = cv2.VideoCapture(str(self.source))
        if not self.cap.isOpened():
            return False
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        return True

    def read(self):
        return self.cap.read()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


if __name__ == "__main__":
    from src.detector import UAVDetector
    from src.tracker import UAVTracker
    from src.pipeline import UAVTrackingPipeline

    source = sys.argv[1] if len(sys.argv) > 1 else "data/gettyimages-1345327468-640_adpp.mp4"
    detector = UAVDetector(model_path="yolo26n_merged")
    pipeline = UAVTrackingPipeline(detector=detector, tracker=UAVTracker(detector=detector))
    pipeline.run_video(source, max_frames=15, stream_type="my_drone")
