"""
sitecustomize.py — Apptainer NVML compatibility shim for vLLM 0.18.0.

vLLM 0.18.0 uses a vendored pynvml at vllm/third_party/pynvml (loaded via
import_pynvml() in vllm/utils/import_utils.py). In Apptainer containers, NVML
fails to initialize, so cuda_platform_plugin() catches the NVMLError and returns
None → vLLM falls back to UnspecifiedPlatform (device_type="") → crash.

This file is loaded by Python's site.py at interpreter startup (before any user
code), in every process including multiprocessing.spawn subprocesses. It injects
a mock module into sys.modules['vllm.third_party.pynvml'] so that when
import_pynvml() runs, it gets our mock that makes nvmlInit() succeed and
nvmlDeviceGetCount() return real GPU counts via torch.cuda.

Activated when: PYTHONPATH=/workspace is set (done by APPTAINERENV_PYTHONPATH
in start_multi_gpu.sh and main.sh).
"""

import sys
import types


def _build_pynvml_shim():
    """Build a minimal pynvml mock that satisfies vLLM's CUDA platform detection."""

    m = types.ModuleType("pynvml")
    m.__file__ = __file__
    m.__package__ = "pynvml"

    # ── Exception classes ──────────────────────────────────────────────────────
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

    m.NVMLError = NVMLError
    m.NVMLError_DriverNotLoaded = NVMLError_DriverNotLoaded
    m.NVMLError_LibraryNotFound = NVMLError_LibraryNotFound
    m.NVMLError_Unknown = NVMLError_Unknown
    m.NVMLError_Uninitialized = NVMLError_Uninitialized

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    m.nvmlInit = lambda: None       # pretend NVML initialised
    m.nvmlShutdown = lambda: None

    # ── Device enumeration ─────────────────────────────────────────────────────
    def nvmlDeviceGetCount():
        try:
            import torch
            n = torch.cuda.device_count()
            return n if n > 0 else 2
        except Exception:
            return 2   # ada-lovelace has 2 L40S GPUs
    m.nvmlDeviceGetCount = nvmlDeviceGetCount

    # ── Device handles ─────────────────────────────────────────────────────────
    class _Handle:
        def __init__(self, i):
            self.index = i
        def __repr__(self):
            return f"<NvmlHandle device={self.index}>"

    m.nvmlDeviceGetHandleByIndex = lambda i: _Handle(i)

    # ── Capability queries (delegate to torch.cuda) ────────────────────────────
    def nvmlDeviceGetName(h):
        try:
            import torch
            name = torch.cuda.get_device_name(h.index)
            return name.encode("utf-8") if isinstance(name, str) else name
        except Exception:
            return b"NVIDIA GPU"
    m.nvmlDeviceGetName = nvmlDeviceGetName

    def nvmlDeviceGetCudaComputeCapability(h):
        try:
            import torch
            return torch.cuda.get_device_capability(h.index)
        except Exception:
            return (8, 9)   # Ada Lovelace — L40S / L4
    m.nvmlDeviceGetCudaComputeCapability = nvmlDeviceGetCudaComputeCapability

    class _MemInfo:
        def __init__(self, i):
            try:
                import torch
                free, total = torch.cuda.mem_get_info(i)
                self.total = total
                self.free = free
                self.used = total - free
            except Exception:
                self.total = 48 * 1024 ** 3
                self.free  = 44 * 1024 ** 3
                self.used  =  4 * 1024 ** 3

    m.nvmlDeviceGetMemoryInfo = lambda h: _MemInfo(h.index)
    m.nvmlDeviceGetUUID = lambda h: f"GPU-shim-{h.index:04x}"

    class _PciInfo:
        busId = b"0000:00:00.0"
    m.nvmlDeviceGetPciInfo = lambda h: _PciInfo()

    return m


_shim = _build_pynvml_shim()

# Inject as both the vendored path vLLM uses and the bare 'pynvml' name.
# setdefault so we never overwrite a real module if already imported.
sys.modules.setdefault("vllm.third_party.pynvml", _shim)
sys.modules.setdefault("pynvml", _shim)
