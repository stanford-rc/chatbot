"""
sitecustomize.py — Apptainer NVML compatibility shims for vLLM 0.18.0.

Loaded by Python's site.py at interpreter startup (before any user code),
in every process including multiprocessing.spawn subprocesses. Activated
when PYTHONPATH=/workspace is set via APPTAINERENV_PYTHONPATH.

Shim 1 — pynvml mock
  vLLM 0.18.0 uses a vendored pynvml at vllm/third_party/pynvml (loaded via
  import_pynvml() in vllm/utils/import_utils.py). In Apptainer, NVML fails
  to initialize → cuda_platform_plugin() returns None → UnspecifiedPlatform
  (device_type="") → "Device string must not be empty" crash.
  Fix: inject mock modules for vllm.third_party, vllm.third_party.pynvml,
  and pynvml so nvmlInit() succeeds and GPU counts come from torch.cuda.

Shim 2 — GPUWorker.determine_available_memory patch (import hook)
  During the profile run, AWQ→Marlin weight conversion allocates temporary
  tensors (~16–25 GiB) that PyTorch's caching allocator retains after they
  are freed. torch.cuda.mem_get_info() sees the cache as "used" at the CUDA
  driver level, so vLLM calculates only ~0.48 GiB available for KV cache on
  a 22.5 GiB L4 with an 18 GiB model — clearly wrong.
  Fix: intercept the import of vllm.v1.worker.gpu_worker and wrap
  GPUWorker.determine_available_memory to call torch.cuda.empty_cache()
  before measuring free memory, releasing the allocator cache to the driver.

⚠ FUTURE ISSUE — NVML / Apptainer:
  Even with the host NVIDIA driver correctly installed, NVML (nvidia-ml
  library) may not initialise inside Apptainer containers.  This causes:
  • vLLM platform detection to fail (fixed by Shim 1 above)
  • PyTorch's CUDACachingAllocator to assert nvmlInit_v2_() at
    CUDACachingAllocator.cpp:1124, killing CUDA graph capture
    (worked around by enforce_eager=True when it occurs)
  • torch.cuda.mem_get_info() returning inaccurate values (fixed by Shim 2)
  The root cause is a driver/library version mismatch or the NVML management
  API being blocked inside the container namespace.  Symptoms will reappear
  after any host driver update or OS upgrade on ada-lovelace until the
  Apptainer NVML binding is confirmed to match.  Verify with:
    nvidia-smi   (on host — must succeed with no version-mismatch error)
    apptainer exec --nv <sif> python -c "import pynvml; pynvml.nvmlInit()"
"""

import sys
import types


# ── Shim 1: pynvml mock ────────────────────────────────────────────────────

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
            return 2   # ada-lovelace has 2 L4 GPUs
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
            return (8, 9)   # Ada Lovelace — NVIDIA L4
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
                # NVIDIA L4: 22.5 GiB (23,034 MiB) per GPU
                self.total = 23034 * 1024 ** 2
                self.free  = 21000 * 1024 ** 2
                self.used  =  2034 * 1024 ** 2

    m.nvmlDeviceGetMemoryInfo = lambda h: _MemInfo(h.index)
    m.nvmlDeviceGetUUID = lambda h: f"GPU-shim-{h.index:04x}"

    class _PciInfo:
        busId = b"0000:00:00.0"
    m.nvmlDeviceGetPciInfo = lambda h: _PciInfo()

    return m


_shim = _build_pynvml_shim()

# vllm.third_party does NOT exist as a real package in the ARM64+cu130 vLLM
# 0.18.0 wheel. Python's import system requires every component of a dotted
# path to be resolvable before the leaf. If 'vllm.third_party' isn't in
# sys.modules, `import vllm.third_party.pynvml` triggers a filesystem lookup
# for vllm/third_party/__init__.py — which doesn't exist — and raises:
#   ImportError: cannot import name 'third_party' from 'vllm'
# Fix: inject a fake package for the parent so Python skips the filesystem.
_third_party_mod = types.ModuleType("vllm.third_party")
_third_party_mod.__path__ = []          # presence of __path__ (even empty) marks it as a package
_third_party_mod.__package__ = "vllm.third_party"
_third_party_mod.pynvml = _shim        # convenience attribute

# Inject as both the vendored path vLLM uses and the bare 'pynvml' name.
# setdefault so we never overwrite a real module if already imported.
sys.modules.setdefault("vllm.third_party", _third_party_mod)
sys.modules.setdefault("vllm.third_party.pynvml", _shim)
sys.modules.setdefault("pynvml", _shim)


# ── Shim 2: GPUWorker.determine_available_memory patch ────────────────────

import importlib.abc as _abc
import importlib.util as _iutil


class _GpuWorkerPatcher(_abc.MetaPathFinder):
    """Import hook that wraps GPUWorker.determine_available_memory.

    Root cause: after the AWQ→Marlin weight conversion during vLLM's profile
    run, PyTorch's CUDA caching allocator retains freed tensors from the
    original AWQ weights. torch.cuda.mem_get_info() (CUDA driver level) sees
    this cache as "used", so determine_available_memory() reports ~0.48 GiB
    free on a 22.5 GiB L4 with an 18 GiB model — leading to ValueError in
    _check_enough_kv_cache_memory.

    Fix: call torch.cuda.empty_cache() before the measurement so the caching
    allocator releases its hold and the driver sees the true free memory (~26 GiB).
    """
    _TARGET = 'vllm.v1.worker.gpu_worker'
    _patched = False

    def find_spec(self, fullname, path, target=None):
        if self._patched or fullname != self._TARGET:
            return None
        # Temporarily remove self to prevent infinite recursion when
        # find_spec is re-entered while we look up the real spec.
        sys.meta_path.remove(self)
        try:
            spec = _iutil.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None:
            return None

        original_loader = spec.loader
        patcher = self

        class _Loader(_abc.Loader):
            def create_module(self, s):
                if hasattr(original_loader, 'create_module'):
                    return original_loader.create_module(s)
                return None

            def exec_module(self, mod):
                original_loader.exec_module(mod)
                patcher._apply(mod)
                patcher._patched = True

        spec.loader = _Loader()
        return spec

    @staticmethod
    def _apply(mod):
        cls = getattr(mod, 'GPUWorker', None)
        if cls is None:
            return
        orig = cls.determine_available_memory

        def _patched_determine_available_memory(self):
            import torch
            # Flush PyTorch's CUDA caching allocator so that memory freed
            # during the profile pass (AWQ tensors replaced by Marlin tensors,
            # activation buffers, etc.) is returned to the CUDA driver before
            # mem_get_info() reads the free-memory counter.  Without this,
            # ~25 GiB of cached-but-freed tensors show as "used" and vLLM
            # calculates near-zero KV cache headroom on a 22.5 GiB L4.
            torch.cuda.empty_cache()
            return orig(self)

        cls.determine_available_memory = _patched_determine_available_memory


sys.meta_path.insert(0, _GpuWorkerPatcher())
