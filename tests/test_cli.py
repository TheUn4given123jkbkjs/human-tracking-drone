"""
Unit test kiểm thử tính năng CLI cơ bản (module nạp động và tạo file MP4).
"""

import sys
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

def test_custom_receiver_loading():
    cmd = [
        sys.executable,
        "track_drone.py",
        "--model", "yolo26n_merged",
        "--receiver-module", "examples/custom_drone_receiver.py",
        "--stream-type", "my_drone",
        "--max-frames", "10"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Lỗi thực thi CLI: {res.stderr}"
    assert "Đã nạp thành công module receiver" in res.stdout, "Không tìm thấy log nạp module"
    print("  --> PASS: test_custom_receiver_loading thành công.")

def test_cli_output_record():
    out_test = ROOT / "runs/test_cli_out.mp4"
    if out_test.exists():
        out_test.unlink()

    cmd = [
        sys.executable,
        "track_drone.py",
        "--model", "yolo26n_merged",
        "--out", str(out_test),
        "--record-fps", "25.0",
        "--max-frames", "10"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Lỗi thực thi CLI: {res.stderr}"
    assert out_test.exists() and out_test.stat().st_size > 1000, "File video đầu ra không hợp lệ"
    out_test.unlink()
    print("  --> PASS: test_cli_output_record thành công.")

if __name__ == "__main__":
    test_custom_receiver_loading()
    test_cli_output_record()
