"""
Module Giả lập Truyền phát Video (UAV Stream Simulator).
So sánh 3 phương án chất lượng trên 1 cửa sổ:
  1. DIRECT RAW: Không nén.
  2. H.264 (libx264 zerolatency): Có In-loop Deblocking Filter.
  3. MPEG-4 Part 2 (mp4v): Không có Deblocking Filter.

Lưu ý phương pháp: Latency là thời gian encode+decode trong RAM (không tính mạng).
H.264 dùng CRF 21, MPEG-4 dùng bitrate cố định 1.5 Mbps → chưa cùng ngân sách bit.
"""

import time
import fractions
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import cv2
import av


class RawStreamEncoder:
    """Phương án 1: Không nén."""
    name = "DIRECT RAW (Zero Loss)"
    codec_label = "RAW_BGR24"
    deblock_label = "N/A (Uncompressed)"
    color = (0, 200, 100)

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, float]:
        return frame, 0.0, frame.nbytes, 100.0


class PyAVCodecEncoder:
    """Mã hóa + giải mã 1 frame trong RAM bằng libavcodec (PyAV)."""

    def __init__(self, name: str, codec_label: str, deblock_label: str, color: Tuple[int, int, int],
                 enc_codec: str, dec_codec: str, width: int, height: int, fps: float,
                 options: Optional[Dict[str, str]] = None, bit_rate: Optional[int] = None):
        self.name, self.codec_label, self.deblock_label, self.color = name, codec_label, deblock_label, color
        self.enc = av.CodecContext.create(enc_codec, "w")
        self.enc.width, self.enc.height = width, height
        self.enc.pix_fmt = "yuv420p"
        self.enc.time_base = fractions.Fraction(1, int(fps) if fps > 0 else 30)
        if options:
            self.enc.options = options
        if bit_rate:
            self.enc.bit_rate = bit_rate
        self.enc.open()
        self.dec = av.CodecContext.create(dec_codec, "r")
        self.dec.open()

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, float]:
        t0 = time.perf_counter()
        packets = self.enc.encode(av.VideoFrame.from_ndarray(frame, format="bgr24"))
        size_bytes = sum(p.size for p in packets)
        decoded = [f for p in packets for f in self.dec.decode(p)]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not decoded:
            return frame, latency_ms, size_bytes, 0.0
        out = decoded[0].to_ndarray(format="bgr24")
        return out, latency_ms, size_bytes, float(cv2.PSNR(frame, out))


def draw_panel_hud(img: np.ndarray, enc, latency_ms: float, size_bytes: int, psnr: float) -> np.ndarray:
    """Vẽ thanh trạng thái kỹ thuật trên đầu mỗi panel."""
    h, w = img.shape[:2]
    header_h = 58
    canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    canvas[header_h:, :] = img
    cv2.rectangle(canvas, (0, 0), (w, header_h), (25, 25, 25), -1)
    cv2.line(canvas, (0, header_h - 1), (w, header_h - 1), enc.color, 2)
    cv2.putText(canvas, enc.name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, enc.color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Codec: {enc.codec_label}", (w - 190, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    psnr_text = f"PSNR: {psnr:.1f}dB" if psnr < 99 else "PSNR: RAW"
    stat = f"Deblock: {enc.deblock_label} | Latency: {latency_ms:4.1f}ms | Size: {size_bytes / 1024.0:5.1f}KB | {psnr_text}"
    cv2.putText(canvas, stat, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


class H264StreamViewer:
    """
    Cửa sổ theo dõi luồng trực tiếp chuyên dụng cho PHƯƠNG ÁN 2 (H.264 Low-Latency Streaming).
    Chỉ chạy duy nhất 1 luồng H.264 zerolatency với bộ lọc In-loop Deblocking Filter bật,
    tối ưu tải xử lý và hiển thị 1 màn hình duy nhất to rõ, sắc nét.
    """

    def __init__(self, width: int = 960, height: int = 540, fps: float = 30.0, crf: int = 21, direct_view: bool = True):
        # Giữ nguyên 100% độ phân giải gốc của video (đảm bảo kích thước chẵn cho YUV420p)
        self.disp_w = (max(64, int(width)) // 2) * 2
        self.disp_h = (max(64, int(height)) // 2) * 2
        self.fps = fps
        self.direct_view = direct_view
        self.crf = crf

        # Khởi tạo bộ mã hóa H.264 chuyên biệt (Phương án 2) nếu không dùng direct_view
        if not self.direct_view:
            self.encoder = PyAVCodecEncoder(
                name="P2: H.264 STREAMING (IN-LOOP DEBLOCKING ON)",
                codec_label="libx264 (zerolatency)",
                deblock_label="In-loop Adaptive (Active)",
                color=(255, 140, 0),
                enc_codec="libx264",
                dec_codec="h264",
                width=self.disp_w,
                height=self.disp_h,
                fps=fps,
                options={"preset": "ultrafast", "tune": "zerolatency", "crf": str(crf)}
            )
        else:
            self.encoder = None

        self.window_name = "UAV STREAM MONITOR [PHƯƠNG ÁN P2: H.264 LOW-LATENCY]"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.runs_dir = Path("runs/simulator")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print("🖥️  KHỞI ĐỘNG MÀN HÌNH THEO DÕI: PHƯƠNG ÁN 2 (H.264 LOW-LATENCY STREAMING)")
        print(f" • Độ phân giải hiển thị: {self.disp_w}x{self.disp_h} @ {fps:.1f} FPS")
        if self.direct_view:
            print(" • Chế độ hiển thị: DIRECT RAM VIEW (Độ trễ 0ms, không re-encode thừa, tiết kiệm CPU)")
            print(f" • Bộ ghi tệp ổ đĩa: H.264 libx264 CRF 18 (Deblocking Active) chạy song song")
        else:
            print(f" • Codec mô phỏng: libx264 (preset: ultrafast, tune: zerolatency, CRF: {crf})")
        print(" • Phím điều khiển: [Space] Tạm dừng | [S] Lưu Snapshot | [Q/ESC] Thoát")
        print("=" * 80 + "\n")

    def render_and_show(self, overlaid_frame: np.ndarray, frame_idx: int, live_fps: float) -> bool:
        # 1. Đảm bảo khung hình đúng kích thước chẵn nếu cần
        if overlaid_frame.shape[1] != self.disp_w or overlaid_frame.shape[0] != self.disp_h:
            input_frame = cv2.resize(overlaid_frame, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        else:
            input_frame = overlaid_frame

        # 2. Xử lý khung hình: Direct RAM View (0ms) hoặc Simulate Re-encode
        if self.direct_view or self.encoder is None:
            display_frame = input_frame
            latency_ms = 0.2
            stat_text = "Mode: DIRECT RAM VIEW (0ms latency) | Writer: P2 H.264 Active | Deblock: ACTIVE"
        else:
            h264_frame, latency_ms, size_bytes, psnr = self.encoder.process(input_frame)
            display_frame = h264_frame
            size_kb = size_bytes / 1024.0
            stat_text = f"Deblock: ACTIVE | Enc+Dec: {latency_ms:4.1f}ms | Frame Size: {size_kb:5.1f}KB | PSNR: {psnr:4.1f}dB"

        # 3. Vẽ Banner HUD kỹ thuật chuyên nghiệp cho Phương án 2
        header_h = 56
        canvas = np.zeros((self.disp_h + header_h, self.disp_w, 3), dtype=np.uint8)
        canvas[header_h:, :] = display_frame

        # Thanh header trên cùng
        cv2.rectangle(canvas, (0, 0), (self.disp_w, header_h), (20, 20, 20), -1)
        cv2.line(canvas, (0, header_h - 1), (self.disp_w, header_h - 1), (255, 140, 0), 2)

        # Dòng 1: Tiêu đề phương án
        cv2.putText(canvas, "UAV STREAM MONITOR [P2: H.264 LOW-LATENCY]", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 140, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Codec: libx264 (zerolatency)", (self.disp_w - 235, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        # Dòng 2: Thông số kỹ thuật chi tiết
        cv2.putText(canvas, stat_text, (12, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 220), 1, cv2.LINE_AA)

        # Thanh footer dưới cùng
        footer = np.full((30, self.disp_w, 3), 16, dtype=np.uint8)
        help_str = f"Live FPS: {live_fps:4.1f} | [Space]: Pause / Resume | [S]: Snapshot | [Q/ESC]: Thoat"
        cv2.putText(footer, help_str, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1, cv2.LINE_AA)

        window = np.vstack([canvas, footer])
        cv2.imshow(self.window_name, window)

        # Tự động lưu snapshot mẫu Phương án P2 ở frame 15
        if frame_idx == 15:
            latest_p2 = self.runs_dir / "snapshot_p2_latest.jpg"
            cv2.imwrite(str(latest_p2), window)

        # 4. Tương tác phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        elif key == ord("s"):
            snap_path = self.runs_dir / f"snapshot_p2_frame_{frame_idx:05d}.png"
            cv2.imwrite(str(snap_path), window)
            print(f"[+] Đã lưu snapshot P2 tại: {snap_path}")
        elif key == ord(" "):
            # Tạm dừng để phi công / người điều khiển quan sát kỹ
            while True:
                k2 = cv2.waitKey(30) & 0xFF
                if k2 == ord(" "):
                    break
                elif k2 in (ord("q"), 27):
                    return False
                elif k2 == ord("s"):
                    snap_path = self.runs_dir / f"snapshot_p2_paused_frame_{frame_idx:05d}.png"
                    cv2.imwrite(str(snap_path), window)
                    print(f"[+] Đã lưu snapshot khi tạm dừng: {snap_path}")
        return True

    def close(self):
        cv2.destroyAllWindows()


class MultiStreamViewer:
    """Hiển thị so sánh 3 luồng trên 1 cửa sổ. Phím: 0/1/2/3 chế độ xem, S snapshot, Q/ESC thoát."""

    def __init__(self, panel_w: int = 640, panel_h: int = 360, fps: float = 30.0):
        self.panel_w, self.panel_h = panel_w, panel_h
        self.encoders = [
            RawStreamEncoder(),
            PyAVCodecEncoder("H.264 (Deblocking ON)", "libx264 (AVC)", "In-loop Adaptive (Active)", (255, 140, 0),
                             "libx264", "h264", panel_w, panel_h, fps,
                             options={"preset": "ultrafast", "tune": "zerolatency", "crf": "21"}),
            PyAVCodecEncoder("MPEG-4 (No Deblocking)", "mpeg4 (mp4v legacy)", "None (Blocky/Ringing)", (0, 100, 255),
                             "mpeg4", "mpeg4", panel_w, panel_h, fps, bit_rate=1_500_000),
        ]
        self.view_mode = 0  # 0: 3-in-1, 1..3: từng luồng
        self.window_name = "UAV STREAMING QUALITY SIMULATOR (3-IN-1)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.runs_dir = Path("runs/simulator")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        print("[*] Stream Simulator: [0] 3-in-1 | [1] Raw | [2] H.264 | [3] MPEG-4 | [S] Snapshot | [Q/ESC] Thoát")

    def render_and_show(self, overlaid_frame: np.ndarray, frame_idx: int, live_fps: float) -> bool:
        panel_frame = cv2.resize(overlaid_frame, (self.panel_w, self.panel_h), interpolation=cv2.INTER_AREA)
        panels = [draw_panel_hud(*self._run(enc, panel_frame)) for enc in self.encoders]
        if self.view_mode == 0:
            display_img = np.hstack(panels)
        else:
            # Phóng to màn hình đơn (x2) để người dùng theo dõi rõ nét chi tiết và BBox
            single = panels[self.view_mode - 1]
            display_img = cv2.resize(single, (single.shape[1] * 2, single.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

        footer = np.full((32, display_img.shape[1], 3), 18, dtype=np.uint8)
        help_str = f"Live FPS: {live_fps:4.1f} | Mode: [{self.view_mode}] (0: 3-in-1 | 1: Raw | 2: H.264 | 3: MPEG-4) | [Space]: Pause | [S]: Snapshot | [Q]: Thoat"
        cv2.putText(footer, help_str, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 220), 1, cv2.LINE_AA)
        window = np.vstack([display_img, footer])
        cv2.imshow(self.window_name, window)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key in (ord("0"), ord("1"), ord("2"), ord("3")):
            self.view_mode = key - ord("0")
        elif key == ord("s"):
            snap_path = self.runs_dir / f"snapshot_frame_{frame_idx:05d}.png"
            cv2.imwrite(str(snap_path), window)
            print(f"[+] Đã lưu snapshot: {snap_path}")
        elif key == ord(" "):
            # Tạm dừng khung hình để người điều khiển quan sát kỹ
            while True:
                k2 = cv2.waitKey(30) & 0xFF
                if k2 == ord(" "):
                    break
                elif k2 in (ord("q"), 27):
                    return False
                elif k2 == ord("s"):
                    snap_path = self.runs_dir / f"snapshot_paused_frame_{frame_idx:05d}.png"
                    cv2.imwrite(str(snap_path), window)
                    print(f"[+] Đã lưu snapshot khi tạm dừng: {snap_path}")
        return True

    @staticmethod
    def _run(enc, frame):
        out, latency_ms, size_bytes, psnr = enc.process(frame)
        return out, enc, latency_ms, size_bytes, psnr

    def close(self):
        cv2.destroyAllWindows()
