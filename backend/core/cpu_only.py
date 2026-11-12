"""CPU-only device manager for the ``feat/cpu-lightweight`` branch.

One module owns every hardware decision this branch makes:

* **Device** - always ``cpu``. ``core.device_caps`` (the app's device choke
  point) and ``services.model_manager.get_best_device`` both defer to
  :func:`enabled`, and :func:`apply_environment` hides GPUs from the whole
  process tree, so engine sidecars that probe ``torch.cuda`` themselves (they
  run in their own venvs and never import the backend) see none either.
* **Precision** - float32 (:func:`dtype_name`); half precision has no fast CPU
  GEMM and some Half kernels are unimplemented on CPU.
* **Threads** - sized from *physical* cores (:func:`thread_plan`) so a
  dual/quad-core machine is not oversubscribed by hyper-threads or by several
  runtimes (torch, OpenMP, MKL, ONNX Runtime) each claiming every core.
* **Memory** - :func:`release_memory` (``gc`` + returning freed heap to the
  OS) replaces accelerator cache flushes; :func:`ram_snapshot` replaces VRAM
  monitoring; :func:`torch_load` memory-maps checkpoints.
* **Low-resource mode** - :func:`low_resource_mode` (on by default) lowers
  batch widths to 1, stops startup preloads, and unloads idle models sooner.

Stdlib-only at import (``psutil`` and ``torch`` are imported lazily) so it can
run before anything heavy is loaded, and nothing here ever raises to a caller.
"""
from __future__ import annotations

import ctypes
import gc
import logging
import os
import sys

logger = logging.getLogger("omnivoice.cpu_only")

#: This branch is CPU-only. Not a setting: there is no supported way to turn
#: it off in a running app.
CPU_ONLY = True

#: Test-only switch. The GPU routing code is kept (dormant) so the branch can
#: keep merging ``main``; its existing tests mock accelerators and need the
#: policy out of the way. Honoured only inside a pytest run (the test process
#: or a child it spawned).
_TEST_ESCAPE_ENV = "OMNIVOICE_TEST_ALLOW_GPU_PATHS"

_LOW_RESOURCE_ENV = "OMNIVOICE_LOW_RESOURCE_MODE"
_LOW_RESOURCE_PREF = "low_resource_mode"
_FALSE = {"0", "false", "no", "off"}

#: Variables that make every GPU runtime report "no device". An empty value is
#: the documented way to hide all CUDA / ROCm devices from a process.
_HIDE_GPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "",
    # The app's own pin: honoured by device_caps, the gguf engines and the
    # sidecars that select a device themselves.
    "OMNIVOICE_DEVICE": "cpu",
    "OMNIVOICE_TORCH_VARIANT": "cpu",
    # Sidecar engines' own device / precision switches.
    "OMNIVOICE_VOXCPM_DEVICE": "cpu",
    "OMNIVOICE_INDEXTTS_FP16": "0",
    "OMNIVOICE_DOTS_TTS_PRECISION": "float32",
}
#: Removed outright: it asks for CUDA even when the probe says no.
_DROP_ENV = ("OMNIVOICE_FORCE_CUDA",)

#: Thread-count variables read by the native math runtimes at load time.
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def enabled() -> bool:
    """Whether the CPU-only policy applies. Always true outside pytest."""
    if not CPU_ONLY:
        return False
    # PYTEST_CURRENT_TEST covers helper processes a test spawns (ASR sidecar).
    under_pytest = "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ
    if under_pytest and os.environ.get(_TEST_ESCAPE_ENV, "").strip() == "1":
        return False
    return True


def low_resource_mode() -> bool:
    """The ``low_resource_mode`` flag: env > prefs.json > ``True``."""
    raw = os.environ.get(_LOW_RESOURCE_ENV, "").strip().lower()
    if raw:
        return raw not in _FALSE
    try:
        from core import prefs

        value = prefs.get(_LOW_RESOURCE_PREF, True)
    except Exception:  # noqa: BLE001 - prefs unreadable: keep the safe default
        return True
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE
    return bool(value)


# ── threads ──────────────────────────────────────────────────────────────


def physical_cores() -> int:
    """Physical cores this process may use (never below 1).

    Hyper-threads share one FPU/SIMD unit, so sizing a matrix-multiply pool
    from logical CPUs makes two threads fight over each core. Also capped by
    the process affinity mask so a container limited to 2 CPUs on a 64-core
    host does not start 32 threads.
    """
    logical = os.cpu_count() or 1
    physical = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001 - psutil missing or unsupported platform
        physical = None
    if not physical:
        physical = max(1, logical // 2) if logical > 2 else logical
    try:
        allowed = len(os.sched_getaffinity(0))  # Linux only
    except (AttributeError, OSError):
        allowed = logical
    return max(1, min(int(physical), allowed))


def thread_plan() -> tuple[int, int]:
    """``(intra_op, inter_op)`` thread counts for this host.

    Intra-op (inside one matmul) gets the physical cores. Inter-op (parallel
    independent ops) stays at 1 up to four cores: a second inter-op pool there
    only multiplies threads over the same cores.
    """
    cores = physical_cores()
    override = os.environ.get("OMNIVOICE_CPU_THREADS", "").strip()
    if override.isdigit() and int(override) > 0:
        cores = int(override)
    return cores, (1 if cores <= 4 else 2)


def configure_torch(torch=None) -> tuple[int, int]:
    """Apply :func:`thread_plan` to torch. Safe to call repeatedly."""
    intra, inter = thread_plan()
    try:
        if torch is None:
            import torch  # noqa: PLC0415
        if torch.get_num_threads() != intra:
            torch.set_num_threads(intra)
        try:
            if torch.get_num_interop_threads() != inter:
                torch.set_num_interop_threads(inter)
        except RuntimeError:
            # The inter-op pool is fixed once parallel work has started.
            logger.debug("inter-op thread count already fixed for this process")
    except Exception:  # noqa: BLE001 - thread tuning must never break a load
        logger.debug("torch thread configuration skipped", exc_info=True)
    return intra, inter


# ── environment ──────────────────────────────────────────────────────────


def _low_resource_defaults() -> dict[str, str]:
    """Env defaults applied in low-resource mode. All are existing knobs."""
    return {
        # Batch sizes of 1: flat RAM, no padded batch tensors.
        "OMNIVOICE_DUB_BATCH_WIDTH": "1",
        "OMNIVOICE_NLLB_BATCH_SIZE": "1",
        "OMNIVOICE_GPU_WORKERS": "1",  # the inference pool; one model call at a time
        "OMNIVOICE_CPU_POOL": str(min(2, physical_cores())),
        "OMNIVOICE_LLM_CONCURRENCY": "2",
        # No models loaded until a request needs them.
        "OMNIVOICE_PRELOAD_TTS_ASR": "0",
        "OMNIVOICE_PRELOAD_CAPTURE_ASR": "0",
        "OMNIVOICE_PRELOAD_WATERMARK": "0",
        # Drop idle ASR / side caches before every generate, and unload the
        # idle TTS model after two minutes instead of fifteen.
        "OMNIVOICE_FREE_VRAM_BEFORE_GENERATE": "always",
        "OMNIVOICE_IDLE_TIMEOUT": "120",
    }


def apply_environment(env=None) -> dict:
    """Pin ``env`` (default ``os.environ``) to CPU. Returns the mapping.

    Must run before torch, CTranslate2 or ONNX Runtime are imported: they read
    the device and thread variables once, at load. Children inherit the
    result, which is what makes sidecar engines CPU-only too.

    GPU-hiding variables are overwritten (the policy is not negotiable);
    thread counts and low-resource defaults use ``setdefault`` so an explicit
    user value still wins.
    """
    env = os.environ if env is None else env
    if not enabled():
        return env
    for key, value in _HIDE_GPU_ENV.items():
        env[key] = value
    for key in _DROP_ENV:
        env.pop(key, None)
    intra, _inter = thread_plan()
    for key in _THREAD_ENV:
        env.setdefault(key, str(intra))
    # Tokenizer worker threads on top of the math pools thrash small machines.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if low_resource_mode():
        for key, value in _low_resource_defaults().items():
            env.setdefault(key, value)
    return env


def enforce() -> None:
    """Process entry hook: pin the environment, then log the plan once."""
    apply_environment()
    if enabled():
        intra, inter = thread_plan()
        logger.info(
            "CPU-only build: device=cpu dtype=float32 threads=%d/%d low_resource_mode=%s",
            intra, inter, low_resource_mode(),
        )


# ── device and precision ─────────────────────────────────────────────────


def resolve_device(requested: object = None) -> str:
    """The device string to use. ``requested`` is accepted and ignored."""
    return "cpu"


def device():
    """``torch.device("cpu")``."""
    import torch  # noqa: PLC0415

    return torch.device("cpu")


def dtype_name() -> str:
    """Tensor precision for model weights and activations."""
    return "float32"


# ── memory ───────────────────────────────────────────────────────────────


def ram_snapshot() -> dict:
    """Process and system RAM in MB. Fields are absent when unknown."""
    out: dict = {}
    try:
        import psutil

        info = psutil.Process().memory_info()
        out["rss_mb"] = round(info.rss / (1024 ** 2), 1)
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak is not None:
            out["peak_rss_mb"] = round(peak / (1024 ** 2), 1)
        vm = psutil.virtual_memory()
        out["available_mb"] = round(vm.available / (1024 ** 2), 1)
        out["total_mb"] = round(vm.total / (1024 ** 2), 1)
    except Exception:  # noqa: BLE001 - monitoring is advisory
        pass
    if "peak_rss_mb" not in out:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB, macOS bytes.
            divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
            out["peak_rss_mb"] = round(peak / divisor, 1)
        except Exception:  # noqa: BLE001 - no resource module on Windows
            pass
    return out


def release_memory() -> dict:
    """Free Python garbage and hand freed heap back to the OS.

    The CPU counterpart of ``torch.cuda.empty_cache()``: after ``del model``
    the weights are garbage, but glibc keeps the freed arenas mapped, so RSS
    stays high until ``malloc_trim``. Best-effort; returns a RAM snapshot.
    """
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:  # noqa: BLE001 - musl and friends have no malloc_trim
            pass
    return ram_snapshot()


def torch_load(path, **kwargs):
    """``torch.load`` on CPU with memory-mapped tensor storage.

    Mapping the file means weights are paged in on demand instead of being
    read into a second in-RAM copy during deserialization. Falls back to a
    plain load for legacy (non-zipfile) checkpoints, which cannot be mapped.
    """
    import torch  # noqa: PLC0415

    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", True)
    try:
        return torch.load(path, mmap=True, **kwargs)
    except (RuntimeError, TypeError, ValueError):
        return torch.load(path, **kwargs)


def quantize_dynamic(model):
    """Dynamic int8 quantization of ``nn.Linear`` / ``nn.LSTM`` layers.

    Shrinks those layers' weights about 4x and speeds up CPU matmul. Returns
    the original model unchanged if quantization fails or the host has no
    quantized engine, so callers can apply it unconditionally.
    """
    try:
        import torch  # noqa: PLC0415

        engines = torch.backends.quantized.supported_engines
        if torch.backends.quantized.engine in (None, "none"):
            usable = [e for e in engines if e != "none"]
            if not usable:
                return model
            torch.backends.quantized.engine = usable[-1]
        return torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8,
            inplace=True,  # a copy would hold both models in RAM at once
        )
    except Exception:  # noqa: BLE001 - quantization is an optimization only
        logger.warning("dynamic quantization skipped", exc_info=True)
        return model


def quantization_requested() -> bool:
    """``OMNIVOICE_CPU_QUANTIZE=1`` opts the in-process TTS model into int8."""
    return os.environ.get("OMNIVOICE_CPU_QUANTIZE", "").strip().lower() in {"1", "true", "yes", "on"}
