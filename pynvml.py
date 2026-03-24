"""
pynvml.py — workspace-level shim for Apptainer/Singularity containers.

vLLM 0.18.0 uses pynvml.nvmlInit() to detect the CUDA platform. In Apptainer,
NVML fails silently, causing vLLM to fall back to UnspecifiedPlatform (device_type="")
and crash with "Device string must not be empty".

This shim lives in the working directory (/workspace or the chatbot root) so Python
finds it at sys.path[0] before site-packages — in every process, including vLLM's
spawned EngineCore subprocess.

Design: no top-level imports other than standard library to avoid circular-import
issues during early Python startup. torch is imported lazily inside functions only.
"""


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
    """No-op: pretend NVML initialised successfully."""
    pass

def nvmlShutdown():
    pass


# ── Device enumeration ────────────────────────────────────────────────────────

def nvmlDeviceGetCount() -> int:
    try:
        import torch
        n = torch.cuda.device_count()
        return n if n > 0 else 2
    except Exception:
        return 2  # ada-lovelace has 2 GPUs


# ── Device handles ────────────────────────────────────────────────────────────

class _DeviceHandle:
    def __init__(self, index: int):
        self.index = index
    def __repr__(self) -> str:
        return f"<NvmlHandle device={self.index}>"

def nvmlDeviceGetHandleByIndex(index: int) -> _DeviceHandle:
    return _DeviceHandle(index)


# ── Capability queries ────────────────────────────────────────────────────────

def nvmlDeviceGetName(handle: _DeviceHandle) -> bytes:
    try:
        import torch
        name = torch.cuda.get_device_name(handle.index)
        return name.encode("utf-8") if isinstance(name, str) else name
    except Exception:
        return b"NVIDIA GPU"

def nvmlDeviceGetCudaComputeCapability(handle: _DeviceHandle):
    """Return (major, minor) compute capability — L40/L4 are Ada Lovelace = 8.9."""
    try:
        import torch
        return torch.cuda.get_device_capability(handle.index)
    except Exception:
        return (8, 9)

class _MemoryInfo:
    def __init__(self, index: int):
        try:
            import torch
            free, total = torch.cuda.mem_get_info(index)
            self.total = total
            self.free = free
            self.used = total - free
        except Exception:
            self.total = 48 * 1024 ** 3   # 48 GB (L40S)
            self.free  = 44 * 1024 ** 3
            self.used  =  4 * 1024 ** 3

def nvmlDeviceGetMemoryInfo(handle: _DeviceHandle) -> _MemoryInfo:
    return _MemoryInfo(handle.index)

def nvmlDeviceGetUUID(handle: _DeviceHandle) -> str:
    return f"GPU-apptainer-shim-{handle.index:04x}"

class _PciInfo:
    busId = b"0000:00:00.0"

def nvmlDeviceGetPciInfo(handle: _DeviceHandle) -> _PciInfo:
    return _PciInfo()
