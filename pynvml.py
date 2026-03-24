"""
pynvml.py — workspace-level shim for Apptainer/Singularity containers.

Problem:
  vLLM 0.18.0 uses pynvml.nvmlInit() to detect the CUDA platform. In Apptainer
  containers, NVML fails silently because /proc/driver/nvidia is not accessible.
  This causes vLLM to fall back to UnspecifiedPlatform (device_type = ""), which
  crashes the EngineCore subprocess with NotImplementedError on every platform
  capability check.

  Python-level monkey-patches in the main process don't fix it because vLLM spawns
  EngineCore in a separate process (multiprocessing 'spawn'), which starts fresh.

Solution:
  Place this file in the working directory (/workspace inside the container, the
  same directory that contains app/, config.yaml, etc.). Python searches sys.path
  in order; the working directory comes before site-packages, so this module
  shadows the real pynvml package for ALL processes — main worker and all spawned
  subprocesses — without touching the container image.

  nvmlInit() is faked to succeed. All capability queries delegate to torch.cuda,
  which works correctly in Apptainer even without NVML.

  On non-Apptainer systems where the real pynvml is needed, this file should not
  be in the Python path (i.e., don't run from this directory, or remove the file).
"""

import torch


# ── Exception hierarchy ───────────────────────────────────────────────────────

class NVMLError(Exception):
    pass

class NVMLError_DriverNotLoaded(NVMLError):
    pass

class NVMLError_LibraryNotFound(NVMLError):
    pass

class NVMLError_Unknown(NVMLError):
    pass

class NVMLError_Uninitialized(NVMLError):
    pass


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def nvmlInit():
    """Pretend NVML initialised. torch.cuda handles actual device access."""
    pass

def nvmlShutdown():
    pass


# ── Device enumeration ────────────────────────────────────────────────────────

def nvmlDeviceGetCount() -> int:
    try:
        n = torch.cuda.device_count()
        return n if n > 0 else 2
    except Exception:
        return 2


# ── Device handles ────────────────────────────────────────────────────────────

class _DeviceHandle:
    def __init__(self, index: int):
        self.index = index
    def __repr__(self) -> str:
        return f"<NvmlHandle device={self.index}>"


def nvmlDeviceGetHandleByIndex(index: int) -> _DeviceHandle:
    return _DeviceHandle(index)


# ── Capability queries (delegated to torch.cuda) ──────────────────────────────

def nvmlDeviceGetName(handle: _DeviceHandle) -> bytes:
    try:
        name = torch.cuda.get_device_name(handle.index)
        return name.encode("utf-8") if isinstance(name, str) else name
    except Exception:
        return b"NVIDIA GPU"


def nvmlDeviceGetCudaComputeCapability(handle: _DeviceHandle):
    """Return (major, minor) compute capability tuple."""
    try:
        return torch.cuda.get_device_capability(handle.index)
    except Exception:
        return (8, 9)  # Ada Lovelace (L40 / L4)


class _MemoryInfo:
    def __init__(self, index: int):
        try:
            free, total = torch.cuda.mem_get_info(index)
            self.total = total
            self.free = free
            self.used = total - free
        except Exception:
            self.total = 24 * 1024 ** 3   # 24 GB default (L40 / L4)
            self.free  = 20 * 1024 ** 3
            self.used  =  4 * 1024 ** 3


def nvmlDeviceGetMemoryInfo(handle: _DeviceHandle) -> _MemoryInfo:
    return _MemoryInfo(handle.index)


def nvmlDeviceGetUUID(handle: _DeviceHandle) -> str:
    return f"GPU-apptainer-shim-{handle.index:04x}"


class _PciInfo:
    busId = b"0000:00:00.0"


def nvmlDeviceGetPciInfo(handle: _DeviceHandle) -> _PciInfo:
    return _PciInfo()
