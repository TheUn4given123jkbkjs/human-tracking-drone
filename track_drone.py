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
    print("   python track_drone.py                               # Chạy mặc định với yolo26n_merged (Tối ưu Real-time)")
    print("   python track_drone.py --model yolo26n_merged        # Drone Onboard Model (Mặc định)")
    print("   python track_drone.py --model yolo26s_merged        # Ground Station High-Accuracy Model")
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
        default="data/gettyimages-1345327468-640_adpp.mp4",
        help="Nguồn video đầu vào: đường dẫn file .mp4, camera index (0, 1), hoặc luồng Wi-Fi RTSP drone (rtsp://...)"
    )
    parser.add_argument(
        "--stream-type",
        type=str,
        default="auto",
        help="Loại/Giao thức nguồn video: 'auto' (tự nhận diện), 'camera' (webcam laptop), 'rtsp' (Wi-Fi drone RTSP/UDP), 'file' (tệp video), hoặc tên class đăng ký mới"
    )
    parser.add_argument(
        "--receiver-module",
        type=str,
        default=None,
        help="Đường dẫn file Python chứa class Stream Receiver tùy biến cần nạp (ví dụ: 'examples/custom_drone_receiver.py')"
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=None,
        help="Tốc độ FPS khi ghi video đầu ra (mặc định: tự động theo nguồn hoặc 25 FPS cho live stream)"
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
        default=False,
        help="Hiện/Ẩn vệt đường quỹ đạo tracking (--trail: hiện | --no-trail: ẩn, mặc định: tắt)"
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bật/Tắt tầng Appearance Memory Bank (--memory: bật | --no-memory: tắt)"
    )
    parser.add_argument(
        "--adaptive-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bật/Tắt trích xuất thích ứng theo sự kiện (--adaptive-memory: bật | --no-adaptive-memory: tắt)"
    )
    parser.add_argument(
        "--feat-interval",
        type=int,
        default=3,
        help="Tần suất làm mới đặc trưng định kỳ cho các track ổn định (mỗi N frame, mặc định: 3)"
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
        "--view",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Bật/Tắt cửa sổ theo dõi trực tiếp P2 (--view / --no-view). Mặc định: Tự động BẬT với Camera/RTSP, TẮT với video file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Nạp module receiver tùy biến nếu được chỉ định
    if args.receiver_module:
        import importlib.util
        p = Path(args.receiver_module).resolve()
        if not p.exists():
            raise FileNotFoundError(f"[!] Không tìm thấy file module receiver: {args.receiver_module}")
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"[+] Đã nạp thành công module receiver: {args.receiver_module}")

    # Kiểm tra lệnh hiển thị danh sách models
    if args.list_models:
        print_available_models()
        sys.exit(0)

    # Tự động hóa xử lý khi chỉ định stream-type mà không truyền source
    if args.stream_type == "camera" and args.source == "data/gettyimages-1345327468-640_adpp.mp4":
        args.source = "0"
    elif args.stream_type == "rtsp" and args.source == "data/gettyimages-1345327468-640_adpp.mp4":
        raise ValueError("[!] Khi sử dụng '--stream-type rtsp', bạn cần cung cấp URL luồng drone qua cờ '--source' (ví dụ: --source rtsp://192.168.1.1:8554/live).")

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
        max_lost_frames=args.max_lost_frames,
        adaptive_extraction=args.adaptive_memory,
        feature_update_interval=args.feat_interval
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

    # 5. Tự động hóa thông minh (Smart default) cho cờ --view:
    # - Nếu là live stream (camera 0, rtsp://...): Tự động BẬT cửa sổ theo dõi P2
    # - Nếu là file video: Mặc định chạy nền ghi video, trừ khi người dùng chỉ định --view
    src_str = str(args.source).strip().lower()
    is_live_source = src_str.isdigit() or any(src_str.startswith(p) for p in ["rtsp://", "rtsps://", "udp://", "http://", "https://"])
    should_view = is_live_source if args.view is None else args.view

    # 6. Thực thi bám vết
    pipeline.run_video(
        source_path=args.source,
        output_path=args.out,
        max_frames=args.max_frames,
        view=should_view,
        stream_type=args.stream_type,
        record_fps=args.record_fps
    )


if __name__ == "__main__":
    main()
