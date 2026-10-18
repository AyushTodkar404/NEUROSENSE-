"""
Automatic hardware device detection for NeuroSense.
Scans physical USB ports, Bluetooth Serial (RFCOMM / SPP) ports,
and Bluetooth Low Energy (BLE) for connected BCI hardware.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TypedDict

try:
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import bleak
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False

logger = logging.getLogger("neurosense.detector")


class DeviceInfo(TypedDict):
    connected: bool
    connection_type: str        # "usb", "bluetooth_spp", "bluetooth_ble", "none"
    port: str                   # e.g. "COM3"
    description: str            # e.g. "Standard Serial over Bluetooth link (COM3)"
    manufacturer: str
    hwid: str
    device_type: str            # "NeuroSky MindWave (Bluetooth)", "Arduino (USB)", etc.
    bluetooth_available: bool   # Bluetooth stack present on machine
    bluetooth_ports: list[str]  # All detected Bluetooth COM ports
    usb_ports: list[str]        # All detected USB COM ports


# Global cache for non-blocking BLE scans
_ble_cache: dict = {
    "last_scan": 0.0,
    "devices": [],
    "status": "idle"
}
_ble_lock = threading.Lock()


def _async_ble_scan():
    """Background BLE scan worker that never blocks the main UI thread."""
    if not BLE_AVAILABLE:
        return
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        scanner = bleak.BleakScanner()
        devices = loop.run_until_complete(scanner.discover(timeout=2.0))
        loop.close()
        with _ble_lock:
            _ble_cache["last_scan"] = time.time()
            _ble_cache["devices"] = [(d.name or "Unknown", d.address) for d in devices]
            _ble_cache["status"] = "ok"
    except Exception as exc:
        with _ble_lock:
            _ble_cache["last_scan"] = time.time()
            _ble_cache["status"] = f"error: {exc}"


def trigger_ble_background_scan():
    """Trigger background BLE scan if cache is older than 30 seconds."""
    with _ble_lock:
        if time.time() - _ble_cache["last_scan"] > 30.0 and _ble_cache["status"] != "scanning":
            _ble_cache["status"] = "scanning"
            threading.Thread(target=_async_ble_scan, daemon=True).start()


def detect_eeg_hardware() -> DeviceInfo:
    """
    Automatically scan all laptop USB ports and Bluetooth channels for connected
    EEG hardware, BCI dongles (OpenBCI, NeuroSky), or Arduino microcontrollers.
    """
    if not SERIAL_AVAILABLE:
        return {
            "connected": False,
            "connection_type": "none",
            "port": "",
            "description": "pyserial not available",
            "manufacturer": "",
            "hwid": "",
            "device_type": "none",
            "bluetooth_available": False,
            "bluetooth_ports": [],
            "usb_ports": []
        }

    try:
        all_ports = list(serial.tools.list_ports.comports())
    except Exception as exc:
        logger.warning("Error scanning hardware ports: %s", exc)
        all_ports = []

    usb_ports = []
    bt_ports = []

    for p in all_ports:
        hwid = (p.hwid or "").upper()
        desc = (p.description or "").upper()
        if "BTHENUM" in hwid or "BLUETOOTH" in desc or "BTH" in hwid:
            bt_ports.append(p)
        else:
            usb_ports.append(p)

    # Trigger background BLE check for modern headbands like Muse
    trigger_ble_background_scan()

    # Priority 1: Check for physical USB connected devices (Arduino, OpenBCI Dongle, FTDI)
    usb_keywords = [
        "arduino", "ch340", "cp210", "ftdi", "openbci", "neurosky",
        "usb serial", "usb-serial", "silicon labs", "prolific"
    ]
    for p in usb_ports:
        full_text = f"{p.device} {p.description} {p.manufacturer or ''} {p.hwid or ''}".lower()
        if any(kw in full_text for kw in usb_keywords):
            desc = p.description or p.device
            desc_lower = desc.lower()
            if "openbci" in desc_lower:
                dev_type = "OpenBCI USB Interface"
            elif "neurosky" in desc_lower:
                dev_type = "NeuroSky USB Interface"
            elif "arduino" in desc_lower or "ch340" in desc_lower:
                dev_type = "Arduino USB Bio-Interface"
            else:
                dev_type = "USB Serial BCI Interface"

            return {
                "connected": True,
                "connection_type": "usb",
                "port": p.device,
                "description": desc,
                "manufacturer": p.manufacturer or "",
                "hwid": p.hwid or "",
                "device_type": dev_type,
                "bluetooth_available": len(bt_ports) > 0,
                "bluetooth_ports": [bp.device for bp in bt_ports],
                "usb_ports": [up.device for up in usb_ports]
            }

    # Priority 2: Check for specific Bluetooth BCI keywords (e.g. MindWave, HC-05, TGAM)
    bt_bci_keywords = ["mindwave", "neurosky", "hc-05", "hc-06", "tgam", "bci", "eeg"]
    for p in bt_ports:
        full_text = f"{p.device} {p.description} {p.manufacturer or ''} {p.hwid or ''}".lower()
        if any(kw in full_text for kw in bt_bci_keywords):
            desc = p.description or p.device
            dev_type = "NeuroSky / Bluetooth BCI Headset" if "neurosky" in full_text or "mindwave" in full_text else "Bluetooth Bio-Serial Interface"
            return {
                "connected": True,
                "connection_type": "bluetooth_spp",
                "port": p.device,
                "description": desc,
                "manufacturer": p.manufacturer or "Bluetooth SPP",
                "hwid": p.hwid or "",
                "device_type": dev_type,
                "bluetooth_available": True,
                "bluetooth_ports": [bp.device for bp in bt_ports],
                "usb_ports": [up.device for up in usb_ports]
            }

    # Priority 3: Check BLE cache for discovered Muse / BCI headbands
    with _ble_lock:
        ble_found = list(_ble_cache.get("devices", []))
    for name, addr in ble_found:
        name_lower = name.lower()
        if any(kw in name_lower for kw in ["muse", "openbci", "mindwave", "eeg", "brain"]):
            return {
                "connected": True,
                "connection_type": "bluetooth_ble",
                "port": addr,
                "description": f"{name} ({addr})",
                "manufacturer": "BLE BCI",
                "hwid": addr,
                "device_type": f"{name} (Bluetooth Low Energy)",
                "bluetooth_available": True,
                "bluetooth_ports": [bp.device for bp in bt_ports],
                "usb_ports": [up.device for up in usb_ports]
            }

    # If generic Bluetooth COM ports exist (e.g. COM3, COM4, COM5, COM6)
    bt_port_names = [bp.device for bp in bt_ports]
    usb_port_names = [up.device for up in usb_ports]

    return {
        "connected": False,
        "connection_type": "none",
        "port": bt_port_names[0] if bt_port_names else "",
        "description": f"Bluetooth ready ({len(bt_ports)} ports: {', '.join(bt_port_names)})" if bt_ports else "No USB or Bluetooth serial devices detected",
        "manufacturer": "",
        "hwid": "",
        "device_type": "none",
        "bluetooth_available": len(bt_ports) > 0,
        "bluetooth_ports": bt_port_names,
        "usb_ports": usb_port_names
    }


if __name__ == "__main__":
    result = detect_eeg_hardware()
    print("Auto-detection result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
