import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from acquisition.config import MODE, EEG_PORT, BAUD_RATE
from acquisition.eeg_reader import EEGReader
from acquisition.heart_rate_reader import HeartRateReader
from acquisition.motion_reader import MotionReader
from acquisition.replay_reader import ReplayReader
from acquisition.hardware_reader import HardwareEEGReader
from acquisition.device_detector import detect_eeg_hardware


class UnifiedSensorManager:

    def __init__(self):
        self.mode = MODE.lower()
        self.sampling_rate = 256
        self.window_seconds = 4
        self.step_seconds = 2

        self.window_samples = (
            self.sampling_rate * self.window_seconds
        )

        self.step_samples = (
            self.sampling_rate * self.step_seconds
        )

        self.eeg_reader = EEGReader(sampling_rate=self.sampling_rate)
        self.heart_reader = HeartRateReader()
        self.motion_reader = MotionReader()

        self.replay_reader = None
        self.hardware_reader = None

        if self.mode == "replay":
            self.replay_reader = ReplayReader()
        elif self.mode == "real":
            self.hardware_reader = HardwareEEGReader(
                port=EEG_PORT, baud_rate=BAUD_RATE, sampling_rate=self.sampling_rate
            )

        self.step_samples = (
            self.sampling_rate * self.step_seconds
        )

        self.eeg_buffer = None

    def _update_eeg_window(self, new_eeg):

        new_eeg = np.asarray(
            new_eeg,
            dtype=float
        )

        # First reading: create full 4-second window
        if self.eeg_buffer is None:

            if len(new_eeg) >= self.window_samples:

                self.eeg_buffer = new_eeg[
                    -self.window_samples:
                ]

            else:

                repeats = int(
                    np.ceil(
                        self.window_samples / len(new_eeg)
                    )
                )

                self.eeg_buffer = np.tile(
                    new_eeg,
                    repeats
                )[-self.window_samples:]

        else:

            # Keep the last 2 seconds
            overlap = self.eeg_buffer[
                -self.step_samples:
            ]

            # Add the newest 2 seconds
            latest = new_eeg[
                -self.step_samples:
            ]

            self.eeg_buffer = np.concatenate(
                [
                    overlap,
                    latest
                ]
            )

        return self.eeg_buffer.copy()

    def read_all(self):
        hw_info = detect_eeg_hardware()
        effective_mode = self.mode

        if self.mode == "auto":
            # Auto-detect: if a hardware port is found, switch to real; otherwise simulate
            effective_mode = "real" if hw_info["connected"] else "simulation"

        if effective_mode == "real":
            target_port = hw_info["port"] if hw_info["connected"] else EEG_PORT
            if self.hardware_reader is None or self.hardware_reader.port != target_port:
                self.hardware_reader = HardwareEEGReader(
                    port=target_port, baud_rate=BAUD_RATE, sampling_rate=self.sampling_rate
                )

            try:
                time_axis, new_eeg = self.hardware_reader.read_samples(duration=self.step_seconds)
                eeg = self._update_eeg_window(new_eeg)
                time_axis = np.arange(len(eeg)) / self.sampling_rate
                heart_rate = self.heart_reader.read_heart_rate("Neutral")
                motion_level = self.motion_reader.read_motion("Neutral")
                return {
                    "mode": "real",
                    "hardware_connected": True,
                    "device_type": hw_info.get("device_type", "Generic Serial"),
                    "device_name": hw_info.get("description", target_port),
                    "port": self.hardware_reader.port,
                    "time": time_axis,
                    "eeg": eeg,
                    "heart_rate": heart_rate,
                    "motion_level": motion_level,
                    "hardware_info": hw_info,
                }
            except Exception as exc:
                if self.mode == "auto":
                    # In auto mode, if hardware read fails, gracefully fall back to simulation
                    effective_mode = "simulation"
                else:
                    zeros = np.zeros(self.window_samples)
                    time_axis = np.arange(len(zeros)) / self.sampling_rate
                    return {
                        "mode": "real",
                        "hardware_connected": False,
                        "device_type": "none",
                        "device_name": "Disconnected",
                        "port": target_port,
                        "error": str(exc),
                        "time": time_axis,
                        "eeg": zeros,
                        "heart_rate": 0,
                        "motion_level": 0.0,
                        "hardware_info": hw_info,
                    }

        if effective_mode == "simulation":
            state = getattr(self.eeg_reader, "get_effective_state", lambda: "Neutral")()
            time_axis, new_eeg = (
                self.eeg_reader.generate_eeg(
                    duration=self.step_seconds
                )
            )
            eeg = self._update_eeg_window(new_eeg)
            time_axis = np.arange(len(eeg)) / self.sampling_rate
            heart_rate = self.heart_reader.read_heart_rate(state)
            motion_level = self.motion_reader.read_motion(state)

            return {
                "mode": "simulation",
                "time": time_axis,
                "eeg": eeg,
                "heart_rate": heart_rate,
                "motion_level": motion_level,
                "target_state": state,
                "hardware_connected": hw_info["connected"],
                "connection_type": hw_info.get("connection_type", "none"),
                "device_type": hw_info.get("device_type", "none"),
                "device_name": hw_info.get("description", "None"),
                "port": hw_info.get("port", ""),
                "bluetooth_available": hw_info.get("bluetooth_available", False),
                "bluetooth_ports": hw_info.get("bluetooth_ports", []),
                "hardware_info": hw_info,
            }

        elif effective_mode == "replay":
            data = self.replay_reader.read_next()
            eeg = self._update_eeg_window(data["eeg"])
            time_axis = np.arange(len(eeg)) / self.sampling_rate
            return {
                "mode": "replay",
                "timestamp": data["timestamp"],
                "time": time_axis,
                "eeg": eeg,
                "heart_rate": data["heart_rate"],
                "motion_level": data["motion_level"],
                "hardware_connected": hw_info["connected"],
                "hardware_info": hw_info,
            }

        else:
            raise ValueError(f"Unknown NeuroSense mode: {self.mode}")


if __name__ == "__main__":

    manager = UnifiedSensorManager()

    print("=" * 60)
    print("NeuroSense Rolling Sensor Test")
    print("=" * 60)

    for i in range(3):

        data = manager.read_all()

        print(
            f"\nReading {i + 1}"
        )

        print(
            "Mode:",
            data["mode"]
        )

        print(
            "EEG window:",
            len(data["eeg"]),
            "samples"
        )

        print(
            "Window duration:",
            len(data["eeg"]) / 256,
            "seconds"
        )

        print(
            "Heart rate:",
            data["heart_rate"]
        )

        print(
            "Motion:",
            data["motion_level"]
        )
