"""
Module Giao thức Khóa Mục tiêu (Target Lock Protocol) & Bộ điều phối Bán tự hành (Semi-Autonomous Mission Controller).

Mô phỏng chu trình vận hành thực tế giữa Ground Station (Máy trạm mặt đất) và Onboard Companion Computer (Raspberry Pi 5 / Pixhawk):
1. Giai đoạn 1 (Unconfirmed / All IDs Memory):
   Toàn bộ ID trong tầm nhìn đều được nạp vào Appearance Memory Bank để lưu vết lịch sử ngoại hình.
2. Giai đoạn 2 (Lock Target Packet & ACK):
   Người dùng / Phi công click chọn 1 ID -> Ground Station gửi gói tin `LOCK_TARGET` -> Pi 5 tiếp nhận, khóa template và phản hồi gói `ACK`.
   Sau khi nhận ACK, Memory Bank chuyển sang chế độ Dedicated Memory cho mục tiêu này.
3. Giai đoạn 3 (Semi-Autonomous Flight & Lost Re-acquire):
   Drone bay chế độ bán tự hành (hỗ trợ bám tâm nhưng vẫn cho phép phi công can thiệp lái bằng tay cầm RC).
   Nếu phi công điều khiển làm camera đổi góc nhìn mất dấu mục tiêu:
   -> Drone chuyển sang trạng thái `SPIN_SEARCH` (xoay 360 độ quét tìm kiếm dựa trên Memory Bank).
   -> Nếu đối tượng xuất hiện lại trong góc quét, Re-ID nhận diện lại ngay và chuyển về `TARGET_LOCKED`.
   -> Nếu xoay hết 1 vòng (timeout) không tìm thấy, drone dừng xoay và chuyển sang trạng thái `STANDBY_HOVER` (đứng yên an toàn).
"""

import time
import json
from enum import Enum
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict


class DroneFlightState(str, Enum):
    """Trạng thái điều khiển của UAV trong nhiệm vụ bám vết."""
    SEARCH_ALL_IDS = "SEARCH_ALL_IDS"                  # Tự do: Bật Memory cho toàn bộ ID
    TARGET_LOCKED = "TARGET_LOCKED"                    # Đã khóa: Bán tự hành, bám theo mục tiêu chính
    TARGET_LOST_SPIN_SEARCH = "TARGET_LOST_SPIN_SEARCH"# Mất dấu: Xoay 360 độ quét tìm lại bằng Memory
    STANDBY_HOVER = "STANDBY_HOVER"                    # Hết giờ tìm: Dừng xoay, giữ vị trí an toàn (Hover)


@dataclass
class LockTargetCommand:
    """Gói tin chỉ lệnh khóa mục tiêu từ Ground Station gửi tới Drone (Pi 5)."""
    command_id: str
    target_id: int
    timestamp: float
    mode: str = "SEMI_AUTONOMOUS"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "LockTargetCommand":
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class LockTargetAck:
    """Gói tin phản hồi (ACK) từ Drone (Pi 5) xác nhận đã khóa mục tiêu."""
    command_id: str
    status: str            # "ACK_OK" | "ACK_REJECTED"
    locked_target_id: int
    message: str
    timestamp: float
    initial_bbox: Optional[Tuple[int, int, int, int]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "LockTargetAck":
        data = json.loads(json_str)
        return cls(**data)


class TargetMissionController:
    """
    Bộ điều phối nhiệm vụ và giao thức mô phỏng trên Drone (Pi 5).
    Quản lý máy trạng thái (FSM) và kết nối với Appearance Memory Bank.
    """

    def __init__(
        self,
        memory_bank: Any,
        search_timeout_seconds: float = 12.0,   # Thời gian xoay 360 độ (tương đương 30 deg/s trong 12s)
        yaw_rate_deg_per_sec: float = 30.0
    ):
        self.memory_bank = memory_bank
        self.search_timeout_seconds = search_timeout_seconds
        self.yaw_rate_deg_per_sec = yaw_rate_deg_per_sec

        self.state: DroneFlightState = DroneFlightState.SEARCH_ALL_IDS
        self.locked_target_id: Optional[int] = None
        self.last_seen_target_time: float = time.time()
        self.spin_search_start_time: Optional[float] = None

        print(f"[*] [TargetMissionController] Khởi tạo FSM: Trạng thái ban đầu = {self.state.value}")

    def handle_lock_command(self, cmd_json: str) -> str:
        """
        Nhận gói tin lệnh LOCK_TARGET từ Ground Station và phản hồi gói ACK.
        
        Args:
            cmd_json: Chuỗi JSON của LockTargetCommand.
            
        Returns:
            Chuỗi JSON của LockTargetAck.
        """
        cmd = LockTargetCommand.from_json(cmd_json)
        print(f"\n[GCS -> Pi5] NHẬN LỆNH KHÓA: ID #{cmd.target_id} (Cmd ID: {cmd.command_id})")

        success = self.memory_bank.lock_target(cmd.target_id)
        if success:
            self.locked_target_id = cmd.target_id
            self.state = DroneFlightState.TARGET_LOCKED
            # Chuyển memory bank sang tập trung tối đa vào locked target
            self.memory_bank.mode = "locked_only"
            self.last_seen_target_time = time.time()

            # Lấy bbox hiện tại của target
            target_mem = self.memory_bank.active_tracks.get(cmd.target_id) or self.memory_bank.lost_tracks.get(cmd.target_id)
            bbox = target_mem.last_bbox if target_mem else None

            ack = LockTargetAck(
                command_id=cmd.command_id,
                status="ACK_OK",
                locked_target_id=cmd.target_id,
                message=f"Xác nhận khóa mục tiêu ID #{cmd.target_id}. Khởi động chế độ Bán tự hành.",
                timestamp=time.time(),
                initial_bbox=bbox
            )
            print(f"[Pi5 -> GCS] PHẢN HỒI ACK_OK: Đã khóa ID #{cmd.target_id} thành công!")
        else:
            ack = LockTargetAck(
                command_id=cmd.command_id,
                status="ACK_REJECTED",
                locked_target_id=cmd.target_id,
                message=f"Từ chối: Không tìm thấy ID #{cmd.target_id} trong bộ nhớ.",
                timestamp=time.time()
            )
            print(f"[Pi5 -> GCS] PHẢN HỒI ACK_REJECTED: ID #{cmd.target_id} không tồn tại!")

        return ack.to_json()

    def update_cycle(self, active_track_ids: list, frame_idx: int) -> Dict[str, Any]:
        """
        Cập nhật chu trình FSM theo từng frame video:
        - Giám sát xem locked_target có xuất hiện trong khung hình không.
        - Điều phối chuyển đổi trạng thái (LOCKED -> SPIN_SEARCH -> HOVER).
        
        Returns:
            Dict chứa trạng thái hiện tại, hành động điều khiển UAV (yaw command, hover command).
        """
        now = time.time()
        action = {"action": "NONE", "state": self.state.value, "yaw_rate": 0.0}

        # Nếu chưa khóa mục tiêu, duy trì chế độ tự do
        if self.state == DroneFlightState.SEARCH_ALL_IDS:
            action["action"] = "MANUAL_PILOT_FLIGHT"
            return action

        # Kiểm tra mục tiêu đang khóa có trong frame hiện tại không
        target_in_view = (self.locked_target_id is not None and self.locked_target_id in active_track_ids)

        if self.state == DroneFlightState.TARGET_LOCKED:
            if target_in_view:
                self.last_seen_target_time = now
                action["action"] = "SEMI_AUTONOMOUS_TRACKING"
                action["target_id"] = self.locked_target_id
            else:
                # Mất dấu mục tiêu (do che khuất hoặc phi công đổi góc nhìn)
                print(f"\n[!] [Pi5 FSM] MỤC TIÊU #{self.locked_target_id} MẤT DẤU! KÍCH HOẠT XOAY 360 ĐỘ TÌM KIẾM...")
                self.state = DroneFlightState.TARGET_LOST_SPIN_SEARCH
                self.spin_search_start_time = now
                action["action"] = "INITIATE_360_SPIN_SEARCH"
                action["yaw_rate"] = self.yaw_rate_deg_per_sec

        elif self.state == DroneFlightState.TARGET_LOST_SPIN_SEARCH:
            if target_in_view:
                # TÁI NHẬN DIỆN THÀNH CÔNG NHỜ MEMORY BANK TRONG QUÁ TRÌNH XOAY QUÉT
                print(f"\n[+] [Pi5 FSM] TÁI NHẬN DIỆN THÀNH CÔNG MỤC TIÊU #{self.locked_target_id} TRONG KHI XOAY! DỪNG XOAY, TIẾP TỤC BÁM.")
                self.state = DroneFlightState.TARGET_LOCKED
                self.spin_search_start_time = None
                self.last_seen_target_time = now
                action["action"] = "TARGET_REACQUIRED_RESUME_TRACKING"
                action["target_id"] = self.locked_target_id
                action["yaw_rate"] = 0.0
            else:
                # Đang trong quá trình xoay quét
                elapsed = now - (self.spin_search_start_time or now)
                if elapsed >= self.search_timeout_seconds:
                    # Đã xoay hết 1 vòng (timeout) mà không thấy mục tiêu
                    print(f"\n[!] [Pi5 FSM] XOAY HẾT 1 VÒNG ({self.search_timeout_seconds}s) KHÔNG THẤY MỤC TIÊU #{self.locked_target_id}!")
                    print("[!] [Pi5 FSM] CHUYỂN SANG TRẠNG THÁI STANDBY_HOVER (GIỮ VỊ TRÍ ĐỨNG YÊN AN TOÀN).")
                    self.state = DroneFlightState.STANDBY_HOVER
                    action["action"] = "ENTER_STANDBY_HOVER"
                    action["yaw_rate"] = 0.0
                else:
                    action["action"] = "EXECUTING_360_SPIN_SEARCH"
                    action["yaw_rate"] = self.yaw_rate_deg_per_sec
                    action["search_elapsed"] = round(elapsed, 1)

        elif self.state == DroneFlightState.STANDBY_HOVER:
            if target_in_view:
                # Nếu người đó vô tình đi ngược lại vào khung hình khi drone đang đứng yên
                print(f"[+] [Pi5 FSM] Mục tiêu #{self.locked_target_id} xuất hiện lại khi đang Hover. Tái kích hoạt bám vết!")
                self.state = DroneFlightState.TARGET_LOCKED
                action["action"] = "SEMI_AUTONOMOUS_TRACKING"
            else:
                action["action"] = "HOLD_POSITION_HOVER"

        action["state"] = self.state.value
        return action
