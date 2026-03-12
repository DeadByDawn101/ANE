"""
ane_device.py — ANE device discovery and capability query
Adapted from pyverbs/device.pyx (rdma-core) patterns for Apple Neural Engine.
"""
import ctypes
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional

# Load the ANE bridge dylib
_BRIDGE_PATH = os.path.join(os.path.dirname(__file__), "..", "bridge", "libane_bridge.dylib")

def _load_bridge() -> Optional[ctypes.CDLL]:
    path = os.path.abspath(_BRIDGE_PATH)
    if os.path.exists(path):
        return ctypes.CDLL(path)
    return None

_lib = _load_bridge()


@dataclass
class ANEDeviceInfo:
    """Mirrors ibv_device_attr pattern from libibverbs/device.c"""
    chip: str           # e.g. "M4", "M3 Pro", "M2 Ultra"
    tops_fp16: float    # Theoretical TOPS (FP16)
    sram_mb: int        # ANE on-chip SRAM in MB
    cores: int          # ANE core count
    macos_version: str
    unified_memory_gb: int

    @property
    def tops(self) -> float:
        return self.tops_fp16

    def __str__(self) -> str:
        return (
            f"Apple {self.chip} ANE | "
            f"{self.tops_fp16} TOPS FP16 | "
            f"{self.sram_mb}MB SRAM | "
            f"{self.unified_memory_gb}GB unified memory"
        )


# Known M-series ANE specs (from Apple silicon documentation + ANE benchmarks)
_ANE_SPECS = {
    "M1":       ANEDeviceInfo("M1",       11.0,  16, 16, "", 0),
    "M1 Pro":   ANEDeviceInfo("M1 Pro",   11.0,  16, 16, "", 0),
    "M1 Max":   ANEDeviceInfo("M1 Max",   11.0,  16, 16, "", 0),
    "M1 Ultra": ANEDeviceInfo("M1 Ultra", 22.0,  32, 32, "", 0),
    "M2":       ANEDeviceInfo("M2",       15.8,  16, 16, "", 0),
    "M2 Pro":   ANEDeviceInfo("M2 Pro",   15.8,  16, 16, "", 0),
    "M2 Max":   ANEDeviceInfo("M2 Max",   15.8,  16, 16, "", 0),
    "M2 Ultra": ANEDeviceInfo("M2 Ultra", 31.6,  32, 32, "", 0),
    "M3":       ANEDeviceInfo("M3",       18.0,  16, 16, "", 0),
    "M3 Pro":   ANEDeviceInfo("M3 Pro",   18.0,  16, 16, "", 0),
    "M3 Max":   ANEDeviceInfo("M3 Max",   18.0,  16, 16, "", 0),
    "M3 Ultra": ANEDeviceInfo("M3 Ultra", 36.0,  32, 32, "", 0),
    "M4":       ANEDeviceInfo("M4",       38.0,  16, 16, "", 0),
    "M4 Pro":   ANEDeviceInfo("M4 Pro",   38.0,  16, 16, "", 0),
    "M4 Max":   ANEDeviceInfo("M4 Max",   38.0,  16, 16, "", 0),
    "M4 Ultra": ANEDeviceInfo("M4 Ultra", 76.0,  32, 32, "", 0),
}


class ANEDevice:
    """
    Apple Neural Engine device handle.
    Mirrors ibv_device / ibv_context pattern from libibverbs.
    """

    def __init__(self, info: ANEDeviceInfo):
        self.info = info
        self._lib = _lib

    @classmethod
    def default(cls) -> "ANEDevice":
        """Discover the ANE on this Mac. Raises on non-Apple-Silicon."""
        if platform.system() != "Darwin":
            raise RuntimeError("ANE is only available on macOS (Apple Silicon)")

        chip = cls._detect_chip()
        macos = platform.mac_ver()[0]
        mem_gb = cls._detect_unified_memory_gb()

        spec = _ANE_SPECS.get(chip)
        if spec is None:
            # Unknown chip — use conservative defaults
            spec = ANEDeviceInfo(chip, 15.8, 16, 16, macos, mem_gb)
        else:
            import dataclasses
            spec = dataclasses.replace(spec, macos_version=macos, unified_memory_gb=mem_gb)

        print(f"[ANEDevice] Detected: {spec}")
        return cls(spec)

    @staticmethod
    def _detect_chip() -> str:
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            # e.g. "Apple M4 Pro" → "M4 Pro"
            if "Apple" in out:
                return out.replace("Apple ", "").strip()
        except Exception:
            pass
        # Fallback: check hw.model
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.model"], stderr=subprocess.DEVNULL
            ).decode().strip()
            # e.g. "Mac15,9" — map to chip family (approximate)
            if out.startswith("Mac1"):
                return "M4"
            elif out.startswith("Mac14"):
                return "M3"
        except Exception:
            pass
        return "M-series"

    @staticmethod
    def _detect_unified_memory_gb() -> int:
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL
            ).decode().strip()
            return int(out) // (1024 ** 3)
        except Exception:
            return 0

    @property
    def tops(self) -> float:
        return self.info.tops_fp16

    @property
    def sram_mb(self) -> int:
        return self.info.sram_mb

    @property
    def unified_memory_gb(self) -> int:
        return self.info.unified_memory_gb

    def __repr__(self) -> str:
        return f"ANEDevice({self.info})"
