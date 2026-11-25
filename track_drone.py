"""
=============================================================================
 CLI ENTRYPOINT: UAV HUMAN TRACKING SYSTEM
 Modular Object-Oriented Architecture (src/)
=============================================================================
"""

import sys
import argparse
from pathlib import Path

from src.detector import UAVDetector
from src.tracker import UAVTracker
from src.visualizer import TrackingVisualizer
from src.pipeline import UAVTrackingPipeline
from src.config import (
    DEFAULT_MODEL,
    DEFAULT_TRACKER,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_IMGSZ,
    DEFAULT_DEVICE
)


def print_available_models():
    """In danh sách các mô hình có sẵn trong thư mục models/."""
    print("=" * 80)
    print("🤖 DANH SÁCH MÔ HÌNH HỢP LỆ (QUÉT TỰ ĐỘNG TỪ THƯ MỤC 'models/'):")
    print("=" * 80)
    
    models = UAVDetector.list_models()
    if not models:
        print("  [!] Không tìm thấy mô hình nào chứa weights/best.pt trong 'models/'.")
    else:
        for name, p in models.items():
            size_mb = Path(p).stat().st_size / (1024 * 1024)
            print(f"  • {name:<25} | Kích thước: {size_mb:>5.1f} MB | {p}")
            
    print("-" * 80)
    print("💡 CÁCH GỌI LỆNH CHUẨN XÁC:")
    print("   python track_drone.py --model yolo26s_merged        # Ground Station Model (Khuyên dùng)")
    print("   python track_drone.py --model yolo26n_merged        # Drone Onboard Model")
    print("   python track_drone.py --model yolo26n_visdrone      # Baseline Model")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="UAV Human Tracking System — Modular OOP Architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Tên thư mục mô hình trong models/ (ví dụ: 'yolo26s_merged', 'yolo26n_merged', 'yolo26n_visdrone') hoặc đường dẫn file .pt"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Liệt kê danh sách tất cả các mô hình có sẵn trong 'models/' và thoát"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=DEFAULT_TRACKER,
        choices=["bytetrack", "botsort"],
        help="Thuật toán bám vết: 'bytetrack' | 'botsort'"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/test_drone_10_20m.mp4",
        help="Đường dẫn video đầu vào"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Đường dẫn file hoặc thư mục lưu video kết quả. Mặc định tự sinh vào runs/track/<model>_<tracker>_<video>.mp4"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="Ngưỡng tin cậy phát hiện (Confidence threshold)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help="Ngưỡng IoU cho tracking association"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Kích thước ảnh đầu vào (640, 960, 1280)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Thiết bị tính toán: '0' (GPU) hoặc 'cpu'"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Bước nhảy frame: 1 = Full 30 FPS, 2 = Half-rate 15 FPS"
    )
    parser.add_argument(
        "--trail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hiện/Ẩn vệt đường quỹ đạo tracking (--trail: hiện | --no-trail: ẩn)"
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bật/Tắt tầng Appearance Memory Bank (--memory: bật | --no-memory: tắt)"
    )
    parser.add_argument(
        "--sim-thresh",
        type=float,
        default=0.60,
        help="Ngưỡng Cosine Similarity để Re-ID mục tiêu trong Memory Bank"
    )
    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=60,
        help="Số frame tối đa lưu vết mục tiêu bị mất dấu trong bộ nhớ"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Giới hạn số frame kiểm thử (None = chạy hết video)"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Hiển thị cửa sổ xem trực tiếp"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Kiểm tra lệnh hiển thị danh sách models
    if args.list_models:
        print_available_models()
        sys.exit(0)

    # 1. Khởi tạo UAVDetector (Quét động mô hình theo tên thư mục)
    detector = UAVDetector(
        model_path=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        device=args.device
    )

    # 2. Khởi tạo UAVTracker (Quản lý liên kết quỹ đạo ByteTrack / BoT-SORT + Memory Bank)
    tracker = UAVTracker(
        detector=detector,
        tracker_type=args.tracker,
        use_memory=args.memory,
        similarity_thresh=args.sim_thresh,
        max_lost_frames=args.max_lost_frames
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
