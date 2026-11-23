"""
Module Pipeline tích hợp toàn diện quy trình bám vết người bằng UAV.
"""

import os
import time
from pathlib import Path
from typing import Optional
import cv2

from src.detector import UAVDetector
from src.tracker import UAVTracker
from src.visualizer import TrackingVisualizer
from src.metrics import TrackingMetrics
from src.config import DEFAULT_OUTPUT_DIR


class UAVTrackingPipeline:
    """
    Pipeline điều phối luồng xử lý video từ camera drone:
    Video In -> UAVDetector -> UAVTracker -> TrackingVisualizer -> TrackingMetrics -> Video Out.
    """

    def __init__(
        self,
        detector: UAVDetector,
        tracker: UAVTracker,
        visualizer: Optional[TrackingVisualizer] = None,
        frame_step: int = 1
    ):
        self.detector = detector
        self.tracker = tracker
        self.visualizer = visualizer or TrackingVisualizer()
        self.frame_step = max(1, frame_step)
        self.metrics = TrackingMetrics()

    def run_video(
        self,
        source_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        display: bool = False
    ) -> TrackingMetrics:
        """
        Thực thi bám vết trên toàn bộ video hoặc số frame chỉ định.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"[!] Không tìm thấy video: {source_path}")

        cap = cv2.VideoCapture(source_path)
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Tự động sinh đường dẫn đầu ra
        if output_path is None:
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model_tag = "run2" if "run2" in self.detector.model_path else ("run1" if "run1" in self.detector.model_path else "model")
            vid_stem = Path(source_path).stem
            output_path = str(DEFAULT_OUTPUT_DIR / f"track_{model_tag}_{self.tracker.tracker_type}_{vid_stem}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, orig_fps, (width, height))

        print("="*70)
        print("🚁 UAV HUMAN TRACKING PIPELINE (MODULAR ARCHITECTURE)")
        print(f" • Mô hình:     {self.detector.model_path} ({self.detector.num_parameters:,} params)")
        print(f" • Thuật toán:  {self.tracker.tracker_type.upper()}")
        print(f" • Video vào:   {source_path} ({width}x{height} @ {orig_fps:.1f} FPS, {total_vid_frames} frames)")
        print(f" • Video ra:    {output_path}")
        print(f" • Frame Step:  {self.frame_step} ({'Full 30 FPS' if self.frame_step == 1 else 'Half-rate 15 FPS'})")
        print("="*70)

        self.metrics = TrackingMetrics()
        start_time = time.perf_counter()
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Bỏ qua frame nếu kích hoạt chế độ Half-rate
            if (frame_idx - 1) % self.frame_step != 0:
                continue

            t_start = time.perf_counter()

            # Gọi Tracker
            res = self.tracker.track(frame)

            t_end = time.perf_counter()
            frame_elapsed_ms = (t_end - t_start) * 1000.0

            inf_ms = res.speed.get('inference', 0.0)
            post_ms = res.speed.get('postprocess', 0.0)

            # Vẽ bounding box & trajectory
            frame, active_ids = self.visualizer.draw_tracks(frame, res.boxes)

            # Ghi nhận chỉ số
            self.metrics.record_frame(inf_ms, post_ms, len(active_ids), active_ids)

            # Vẽ HUD
            live_fps = 1000.0 / frame_elapsed_ms if frame_elapsed_ms > 0 else 0.0
            frame = self.visualizer.draw_hud(
                frame=frame,
                model_name=self.detector.model_name,
                tracker_name=self.tracker.tracker_type,
                fps=live_fps,
                inference_ms=inf_ms,
                postprocess_ms=post_ms,
                active_count=len(active_ids),
                unique_total=len(self.metrics.unique_track_ids)
            )

            out.write(frame)

            if display:
                cv2.imshow("UAV Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_idx % 30 == 0 or frame_idx == total_vid_frames or (max_frames and frame_idx == max_frames):
                print(f"[{frame_idx:04d}/{total_vid_frames}] Active: {len(active_ids):02d} | Live FPS: {live_fps:5.1f} | Inf: {inf_ms:4.1f}ms | Post: {post_ms:4.2f}ms")

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        self.metrics.total_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.print_summary(output_path)
        return self.metrics

    def print_summary(self, output_path: str):
        """In báo cáo khoa học tổng hợp sau khi hoàn tất video."""
        m = self.metrics
        print("\n" + "="*70)
        print("📊 BÁO CÁO KHOA HỌC: HIỆU NĂNG BÁM VẾT THỜI GIAN THỰC")
        print("="*70)
        print(f" • Mô hình:                {self.detector.model_name}")
        print(f" • Thuật toán Tracking:     {self.tracker.tracker_type.upper()}")
        print(f" • Tổng số frame xử lý:    {m.total_frames} frames")
        print(f" • Tốc độ trung bình (FPS): {m.average_fps:.2f} FPS")
        print(f" • Độ trễ Inference tb:    {m.avg_inference_ms:.2f} ms")
        print(f" • Hậu xử lý NMS-Free tb:   {m.avg_postprocess_ms:.2f} ms")
        print(f" • Tổng số lượt phát hiện: {m.total_detections} bounding boxes")
        print(f" • Số ID người duy nhất:    {len(m.unique_track_ids)} unique track IDs")
        print(f" • Số người trung bình/frame:{m.avg_active_targets:.2f} targets/frame")
        print(f" • Video kết quả xuất tại:  {output_path}")
        print("="*70)
