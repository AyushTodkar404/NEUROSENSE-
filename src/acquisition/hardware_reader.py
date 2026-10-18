"""
Hardware EEG serial ingest driver for NeuroSense.
Supports physical BCI hardware (OpenBCI, NeuroSky, Arduino/ESP32 ADC frontends)
over USB Serial / Bluetooth Virtual COM ports (e.g. COM3, /dev/ttyUSB0).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
import numpy as np

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

logger = logging.getLogger("neurosense.hardware")


class HardwareEEGReader:
    """
    Ingests live microvolt packets from physical EEG hardware over serial COM ports.
    Features:
    - Auto-detection of available USB / Bluetooth COM ports
    - Non-blocking packet acquisition with rolling buffer
    - Graceful fallback and connection health reporting
    """

    def __init__(self, port: str = "COM3", baud_rate: int = 115200, sampling_rate: int = 256):
        self.port = port
        self.baud_rate = baud_rate
        self.sampling_rate = sampling_rate
        self.ser: serial.Serial | None = None
        self.is_connected = False
        self.last_error = ""
        self._buffer: list[float] = []

    @staticmethod
    def list_available_ports() -> list[str]:
        """List all active hardware COM ports detected by Windows/OS."""
        if not SERIAL_AVAILABLE:
            return []
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            return ports
        except Exception:
            return []

    def connect(self) -> bool:
        """Attempt to open physical serial connection to EEG headset."""
        if not SERIAL_AVAILABLE:
            self.last_error = "pyserial library is not installed in the active environment."
            self.is_connected = False
            return False

        try:
            # Check if requested port exists or attempt first available COM port
            available = self.list_available_ports()
            target_port = self.port
            if target_port not in available and available:
                target_port = available[0]
                logger.info("Configured port %s not found. Using detected port %s", self.port, target_port)

            self.ser = serial.Serial(
                port=target_port,
                baudrate=self.baud_rate,
                timeout=0.1,
                write_timeout=0.1
            )
            self.port = target_port
            self.is_connected = True
            self.last_error = ""
            return True
        except Exception as exc:
            self.is_connected = False
            self.last_error = f"Cannot open hardware port {self.port}: {exc}"
            return False

    def disconnect(self):
        """Close serial port safely."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.is_connected = False

    def read_samples(self, duration: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Read real samples from the connected EEG device.
        Returns (time_axis, eeg_voltages_array).
        """
        n_needed = int(self.sampling_rate * duration)

        if not self.is_connected or self.ser is None or not self.ser.is_open:
            # Attempt reconnect
            if not self.connect():
                # Return empty or baseline signal with error
                raise ConnectionError(self.last_error or f"Hardware not connected on {self.port}")

        # Read available lines from serial
        samples = []
        start_time = time.time()
        timeout = duration + 0.5

        try:
            while len(samples) < n_needed and (time.time() - start_time) < timeout:
                line = self.ser.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    continue
                # Support CSV or single float voltage per line
                parts = line.split(",")
                try:
                    val = float(parts[0])
                    samples.append(val)
                except ValueError:
                    continue
        except Exception as exc:
            self.last_error = f"Serial read error: {exc}"
            self.is_connected = False
            raise ConnectionError(self.last_error)

        if len(samples) < n_needed:
            if not samples:
                raise ConnectionError(f"Zero samples received from {self.port}. Ensure headset is powered on.")
            # Pad with last sample if slight timing jitter
            samples = list(np.pad(samples, (0, n_needed - len(samples)), mode="edge"))

        eeg_array = np.asarray(samples[:n_needed], dtype=float)
        time_axis = np.arange(len(eeg_array)) / float(self.sampling_rate)
        return time_axis, eeg_array


if __name__ == "__main__":
    reader = HardwareEEGReader()
    print("Available hardware ports:", reader.list_available_ports())
