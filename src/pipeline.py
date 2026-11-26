"""
Module Pipeline tích hợp toàn diện quy trình bám vết người bằng UAV.
Hỗ trợ đo lường và hiển thị hiệu năng Appearance Memory Bank.
"""

import time
from pathlib import Path
from typing import Optional, Union
import numpy as np
import cv2

from src.detector import UAVDetector
from src.tracker import UAVTracker
from src.visualizer import TrackingVisualizer
from src.metrics import TrackingMetrics
from src.config import DEFAULT_OUTPUT_DIR
from src.stream_receiver import BaseStreamReceiver, StreamReceiverFactory


class H264VideoWriter:
    """
    Bộ ghi video chuẩn công nghiệp H.264 qua libx264 (PyAV) với In-loop Deblocking Filter bật.
    Giải quyết triệt để hiện tượng vỡ khối hạt (blocking artifact) và suy giảm độ sắc nét của cv2.VideoWriter(mp4v).
    Giữ nguyên 100% độ phân giải gốc và độ chi tiết ở mức Visual Lossless (CRF 18).
    """

    def __init__(self, output_path: str, fps: float, width: int, height: int, crf: int = 18):
        self.output_path = output_path
        self.fps = fps
        self.width = (int(width) // 2) * 2
        self.height = (int(height) // 2) * 2
        self.container = None
        self.stream = None
        self._fallback_cv = None
        self.codec_used = "libx264 (H.264)"

        try:
            import av
            self.container = av.open(output_path, mode="w")
            stream_rate = int(round(fps)) if fps > 0 else 30
            self.stream = self.container.add_stream("libx264", rate=stream_rate)
            self.stream.width = self.width
            self.stream.height = self.height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {
                "preset": "fast",
                "tune": "zerolatency",
                "crf": str(crf)
            }
        except Exception as e:
            # Fallback an toàn sang OpenCV mp4v nếu môi trường thiếu PyAV
            self.codec_used = "mp4v (OpenCV fallback)"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._fallback_cv = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        if self._fallback_cv is not None:
            self._fallback_cv.write(frame)
            return

        import av
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def release(self):
        if self._fallback_cv is not None:
            self._fallback_cv.release()
            self._fallback_cv = None
            return

        if self.stream is not None and self.container is not None:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
                self.container.close()
            except Exception:
                pass
            self.container = None
            self.stream = None


class UAVTrackingPipeline:
    """
    Pipeline điều phối luồng xử lý video từ camera drone:
    Video In -> UAVDetector -> UAVTracker (+Memory Bank) -> TrackingVisualizer -> TrackingMetrics -> Video Out.
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
        source_path: Union[str, int, BaseStreamReceiver],
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        view: bool = False,
        stream_type: Optional[str] = None,
        record_fps: Optional[float] = None
    ) -> TrackingMetrics:
        """
        Thực thi bám vết trên video, camera, luồng Wi-Fi drone hoặc receiver tùy biến.

        Args:
            source_path: Đường dẫn video, chỉ số camera, URL RTSP hoặc instance BaseStreamReceiver.
            output_path: Đường dẫn lưu tệp video kết quả H.264 (tự sinh nếu None).
            max_frames: Giới hạn số frame xử lý.
            view: Bật/Tắt cửa sổ hiển thị theo dõi trực tiếp P2 (H.264 Low-Latency).
            stream_type: 'auto' | 'camera' | 'rtsp' | 'file' | tên receiver đăng ký.
            record_fps: Tốc độ FPS khi ghi video ra file (tùy chọn).
        """
        if isinstance(source_path, BaseStreamReceiver):
            receiver = source_path
        else:
            receiver = StreamReceiverFactory.create(source_path, stream_type=stream_type)
        if not receiver.open():
            raise RuntimeError(f"[!] Không mở được nguồn video: {source_path} ({receiver.__class__.__name__})")

        total_vid_frames = receiver.total_frames or (max_frames or 0)
        orig_fps = receiver.fps
        width = receiver.width
        height = receiver.height

        # Tính FPS ghi video đầu ra (tránh video live bị phát quá nhanh do chênh lệch FPS)
        if record_fps is not None and record_fps > 0:
            writer_fps = float(record_fps)
        elif receiver.is_live:
            writer_fps = min(orig_fps, 25.0)
        else:
            writer_fps = orig_fps

        # Tự động chuẩn hóa đường dẫn đầu ra theo tên model và thuật toán vào runs/track/
        vid_stem = receiver.name
        mem_suffix = "_mem" if getattr(self.tracker, "use_memory", False) else ""
        default_filename = f"{self.detector.model_name}_{self.tracker.tracker_type}{mem_suffix}_{vid_stem}.mp4"

        if output_path is None:
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = str(DEFAULT_OUTPUT_DIR / default_filename)
        else:
            out_p = Path(output_path)
            if out_p.is_dir() or out_p.suffix == "":
                out_p.mkdir(parents=True, exist_ok=True)
                output_path = str(out_p / default_filename)
            else:
                out_p.parent.mkdir(parents=True, exist_ok=True)
                output_path = str(out_p)

        # Sử dụng H264VideoWriter (P2 - libx264 Deblocking Filter) giữ nguyên 100% độ sắc nét gốc
        out = H264VideoWriter(output_path, fps=writer_fps, width=width, height=height, crf=18)

        # Khởi tạo cửa sổ theo dõi P2 nếu kích hoạt cờ view
        viewer = None
        if view:
            from src.stream_simulator import H264StreamViewer
            viewer = H264StreamViewer(width=width, height=height, fps=orig_fps)

        mem_bank = getattr(self.tracker, "memory_bank", None)
        if getattr(self.tracker, "use_memory", False):
            if mem_bank and getattr(mem_bank, "adaptive_extraction", False):
                mem_status = f"BẬT (Appearance Memory Bank - Adaptive Event-Triggered, Interval: {mem_bank.feature_update_interval}f)"
            else:
                mem_status = "BẬT (Appearance Memory Bank - Full Frame)"
        else:
            mem_status = "TẮT"
        print("="*70)
        print("🚁 UAV HUMAN TRACKING PIPELINE (MODULAR ARCHITECTURE)")
        print(f" • Mô hình:     {self.detector.model_name} ({self.detector.num_parameters:,} params)")
        print(f" • Trọng số:    {self.detector.model_path}")
        print(f" • Thuật toán:  {self.tracker.tracker_type.upper()}")
        print(f" • Bộ nhớ Memory: {mem_status}")
        frames_label = f"{total_vid_frames} frames" if total_vid_frames > 0 else "Live Stream"
        print(f" • Nguồn vào:   {source_path} ({width}x{height} @ {orig_fps:.1f} FPS, {frames_label})")
        print(f" • Video ra:    {output_path} (Codec: {out.codec_used}, FPS: {writer_fps:.1f})")
        print(f" • Frame Step:  {self.frame_step} ({'Full 30 FPS' if self.frame_step == 1 else 'Half-rate 15 FPS'})")
        print("="*70)

        self.metrics = TrackingMetrics()
        # Khởi động lại trạng thái tracker và visualizer để bắt đầu ID tuần tự từ 1 cho video mới
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()
        if hasattr(self.visualizer, "reset"):
            self.visualizer.reset()

        start_time = time.perf_counter()
        loop_last_time = start_time
        frame_idx = 0
        live_fps = orig_fps

        last_rendered_frame: Optional[np.ndarray] = None
        signal_lost_start: Optional[float] = None
        last_signal_warn_log: float = 0.0

        try:
            while receiver.is_opened():
                ret, frame = receiver.read()
                if not ret:
                    if not receiver.is_live:
                        # Video file cục bộ đã đọc hết (EOF)
                        break

                    # Luồng trực tiếp (Live Stream / Wi-Fi) mất tín hiệu tạm thời
                    now_ts = time.perf_counter()
                    if signal_lost_start is None:
                        signal_lost_start = now_ts
                        print("\n[!] [UAV Stream] Mất tín hiệu video Wi-Fi! Đang kích hoạt Frame Hold & Auto-Reconnect...")

                    elapsed_lost = now_ts - signal_lost_start

                    # Định kỳ in log cảnh báo mỗi 2.0s
                    if now_ts - last_signal_warn_log >= 2.0:
                        print(f"[*] [UAV Stream] Đang đợi tín hiệu video phục hồi... ({elapsed_lost:.1f}s)")
                        last_signal_warn_log = now_ts

                    # Hiển thị HUD cảnh báo mất sóng trên khung hình cuối cùng (Frame Hold)
                    if viewer is not None and last_rendered_frame is not None:
                        hold_frame = last_rendered_frame.copy()
                        # Làm tối nhẹ 30% để phi công nhận biết trạng thái mất sóng
                        cv2.addWeighted(np.zeros_like(hold_frame), 0.30, hold_frame, 0.70, 0, hold_frame)

                        fh, fw = hold_frame.shape[:2]
                        bw, bh = min(560, fw - 40), 90
                        bx1, by1 = (fw - bw) // 2, (fh - bh) // 2
                        bx2, by2 = bx1 + bw, by1 + bh

                        cv2.rectangle(hold_frame, (bx1, by1), (bx2, by2), (20, 20, 30), -1)
                        cv2.rectangle(hold_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

                        cv2.putText(hold_frame, "SIGNAL LOST - WAITING RECONNECT...", (bx1 + 18, by1 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 50, 255), 2, cv2.LINE_AA)
                        cv2.putText(hold_frame, f"Link Downtime: {elapsed_lost:4.1f}s | Frame Hold Active", (bx1 + 18, by1 + 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
                        cv2.putText(hold_frame, "[Press Q or ESC on Window to Exit]", (bx1 + 18, by1 + 76),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

                        keep_going = viewer.render_and_show(overlaid_frame=hold_frame, frame_idx=frame_idx, live_fps=0.0)
                        if not keep_going:
                            print("[!] [GCS] Phi công yêu cầu dừng phiên theo dõi.")
                            break

                    time.sleep(0.033)  # Giữ nhịp ~30 FPS không gây nghẽn CPU
                    continue

                # Khi có tín hiệu lại sau thời gian mất sóng
                if signal_lost_start is not None:
                    total_lost_duration = time.perf_counter() - signal_lost_start
                    print(f"[✓] [UAV Stream] Tín hiệu video đã khôi phục sau {total_lost_duration:.1f}s! Tiếp tục bám vết.")
                    start_time += total_lost_duration  # Đồng bộ lại dòng thời gian thực
                    signal_lost_start = None

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                # Bỏ qua frame nếu kích hoạt chế độ Half-rate
                if (frame_idx - 1) % self.frame_step != 0:
                    continue

                # Đồng bộ frame_idx theo thời gian vật lý thực khi là live stream
                if receiver.is_live:
                    track_frame_idx = max(frame_idx, int((time.perf_counter() - start_time) * orig_fps) + 1)
                else:
                    track_frame_idx = frame_idx

                t_start = time.perf_counter()

                # Gọi Tracker (kèm track_frame_idx để Memory Bank theo dõi thời gian sống chính xác)
                res = self.tracker.track(frame, frame_idx=track_frame_idx)

                t_end = time.perf_counter()

                inf_ms = res.speed.get('inference', 0.0)
                post_ms = res.speed.get('postprocess', 0.0)

                # Lấy thông tin Re-ID và Locked Target từ kết quả
                reacquired_ids = [p["track_id"] for p in getattr(res, "reacquired_this_frame", [])]
                reacquired_total = 0
                locked_id = None
                if getattr(self.tracker, "use_memory", False) and self.tracker.memory_bank is not None:
                    reacquired_total = self.tracker.memory_bank.total_reacquired_count
                    locked_id = self.tracker.memory_bank.locked_target_id

                # Vẽ bounding box & trajectory
                frame, active_ids = self.visualizer.draw_tracks(
                    frame=frame,
                    boxes_result=res.boxes,
                    reacquired_ids=reacquired_ids,
                    locked_id=locked_id
                )

                # Ghi nhận chỉ số
                self.metrics.record_frame(inf_ms, post_ms, len(active_ids), active_ids)

                # Vẽ HUD (live_fps đo theo toàn bộ chu kỳ vòng lặp End-to-End thực tế)
                frame = self.visualizer.draw_hud(
                    frame=frame,
                    model_name=self.detector.model_name,
                    tracker_name=self.tracker.tracker_type,
                    fps=live_fps,
                    inference_ms=inf_ms,
                    postprocess_ms=post_ms,
                    active_count=len(active_ids),
                    unique_total=len(self.metrics.unique_track_ids),
                    reacquired_total=reacquired_total,
                    locked_id=locked_id
                )
                last_rendered_frame = frame

                out.write(frame)

                # Hiển thị trực tiếp qua cửa sổ theo dõi P2 nếu kích hoạt cờ view
                if viewer is not None:
                    keep_going = viewer.render_and_show(overlaid_frame=frame, frame_idx=frame_idx, live_fps=live_fps)
                    if not keep_going:
                        break

                now_time = time.perf_counter()
                loop_dt = now_time - loop_last_time
                if loop_dt > 0:
                    live_fps = 0.9 * live_fps + 0.1 * (1.0 / loop_dt) if frame_idx > 1 else (1.0 / loop_dt)
                loop_last_time = now_time

                if frame_idx % 30 == 0 or frame_idx == total_vid_frames or (max_frames and frame_idx == max_frames):
                    reid_log = f" | Re-ID: {reacquired_total}" if getattr(self.tracker, "use_memory", False) else ""
                    tag_total = f"{total_vid_frames:04d}" if total_vid_frames > 0 else "LIVE"
                    print(f"[{frame_idx:04d}/{tag_total}] Active: {len(active_ids):02d} | Live FPS: {live_fps:5.1f} | Inf: {inf_ms:4.1f}ms | Post: {post_ms:4.2f}ms{reid_log}")

        finally:
            receiver.release()
            out.release()
            if viewer is not None:
                viewer.close()

        self.metrics.total_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.print_summary(output_path)
        return self.metrics

    def print_summary(self, output_path: str):
        """In báo cáo khoa học tổng hợp sau khi hoàn tất video."""
        m = self.metrics
        mem_bank = getattr(self.tracker, "memory_bank", None)
        reacquired_total = mem_bank.total_reacquired_count if mem_bank is not None else 0

        print("\n" + "="*70)
        print("📊 BÁO CÁO KHOA HỌC: HIỆU NĂNG BÁM VẾT THỜI GIAN THỰC")
        print("="*70)
        print(f" • Mô hình:                {self.detector.model_name}")
        print(f" • Thuật toán Tracking:     {self.tracker.tracker_type.upper()}")
        print(f" • Tầng Appearance Memory:  {'KÍCH HOẠT (Re-ID Bank)' if getattr(self.tracker, 'use_memory', False) else 'TẮT'}")
        if getattr(self.tracker, "use_memory", False) and mem_bank is not None:
            print(f" • Tái nhận diện qua Memory:{reacquired_total} lượt khôi phục ID thành công")
            summary = mem_bank.get_summary()
            total_ops = summary.get("extractions_performed", 0) + summary.get("extractions_skipped", 0)
            if total_ops > 0:
                perf = summary["extractions_performed"]
                skip = summary["extractions_skipped"]
                pct = summary["extractions_skipped_percent"]
                print(f" • Tối ưu CPU Adaptive:    Tiết kiệm {pct:.1f}% lượt trích xuất ({skip:,} bỏ qua / {perf:,} thực hiện)")
        print(f" • Tổng số frame xử lý:    {m.total_frames} frames")
        print(f" • Tốc độ trung bình (FPS): {m.average_fps:.2f} FPS")
        print(f" • Độ trễ Inference tb:    {m.avg_inference_ms:.2f} ms")
        print(f" • Hậu xử lý NMS-Free tb:   {m.avg_postprocess_ms:.2f} ms")
        print(f" • Tổng số lượt phát hiện: {m.total_detections} bounding boxes")
        print(f" • Số ID người duy nhất:    {len(m.unique_track_ids)} unique track IDs")
        print(f" • Số người trung bình/frame:{m.avg_active_targets:.2f} targets/frame")
        print(f" • Video kết quả xuất tại:  {output_path}")
        print("="*70)
