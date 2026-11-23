"""
=============================================================================
 CLI ENTRYPOINT: UAV HUMAN TRACKING SYSTEM
 Modular Object-Oriented Architecture (src/)
=============================================================================
"""

import argparse
from src.detector import UAVDetector
from src.tracker import UAVTracker
from src.visualizer import TrackingVisualizer
from src.pipeline import UAVTrackingPipeline
from src.config import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_IMGSZ, DEFAULT_DEVICE


def parse_args():
    parser = argparse.ArgumentParser(
        description="UAV Human Tracking System — Modular OOP Architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="run2", help="Mô hình weights .pt hoặc alias: 'run1' (VisDrone Baseline) | 'run2' (Merged Run 2)")
    parser.add_argument("--tracker", type=str, default="bytetrack", choices=["bytetrack", "botsort"], help="Thuật toán bám vết: 'bytetrack' | 'botsort'")
    parser.add_argument("--source", type=str, default="data/test_drone_10_20m.mp4", help="Đường dẫn video đầu vào")
    parser.add_argument("--out", type=str, default=None, help="Đường dẫn lưu video kết quả (.mp4)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="IoU threshold cho tracking")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Kích thước ảnh đầu vào (640, 960, 1280)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="Thiết bị: '0' (GPU) hoặc 'cpu'")
    parser.add_argument("--step", type=int, default=1, help="Bước nhảy frame: 1 = Full 30 FPS, 2 = Half-rate 15 FPS")
    parser.add_argument("--trail", action=argparse.BooleanOptionalAction, default=True, help="Hiện/Ẩn vệt đường tracking (--trail: hiện | --no-trail: ẩn)")
    parser.add_argument("--max-frames", type=int, default=None, help="Giới hạn số frame kiểm thử (None = chạy hết video)")
    parser.add_argument("--display", action="store_true", help="Hiển thị cửa sổ xem trực tiếp")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Khởi tạo UAVDetector (Quản lý mô hình và suy luận)
    detector = UAVDetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        device=args.device
    )

    # 2. Khởi tạo UAVTracker (Quản lý liên kết quỹ đạo ByteTrack / BoT-SORT)
    tracker = UAVTracker(
        detector=detector,
        tracker_type=args.tracker
    )

    # 3. Khởi tạo TrackingVisualizer (Vẽ BBox, Trajectory trail, HUD Telemetry)
    visualizer = TrackingVisualizer(max_trail_len=35, show_trail=args.trail)

    # 4. Khởi tạo UAVTrackingPipeline (Điều phối luồng video I/O và số liệu)
    pipeline = UAVTrackingPipeline(
        detector=detector,
        tracker=tracker,
        visualizer=visualizer,
        frame_step=args.step
    )

    # 5. Thực thi bám vết
    pipeline.run_video(
        source_path=args.source,
        output_path=args.out,
        max_frames=args.max_frames,
        display=args.display
    )


if __name__ == "__main__":
    main()
