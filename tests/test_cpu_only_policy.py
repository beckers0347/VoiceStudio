"""The CPU-only branch must resolve every device decision to the CPU, even on
a host where a GPU is present and PyTorch can see it."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import cpu_only, device_caps

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def policy_on(monkeypatch):
    """Run with the policy as the shipped app has it."""
    monkeypatch.delenv("OMNIVOICE_TEST_ALLOW_GPU_PATHS", raising=False)
    monkeypatch.delenv("OMNIVOICE_LOW_RESOURCE_MODE", raising=False)
    device_caps.refresh()
    yield
    device_caps.detect_host_caps.cache_clear()


def test_policy_cannot_be_disabled_outside_pytest():
    code = (
        "import sys; sys.path.insert(0, 'backend'); from core import cpu_only; "
        "print(cpu_only.enabled())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=True,
        env={
            **{k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"},
            "OMNIVOICE_TEST_ALLOW_GPU_PATHS": "1",
        },
    )
    assert out.stdout.strip() == "True"


@pytest.mark.parametrize("requested", ["cuda", "rocm", "mps", "xpu", "auto"])
def test_device_override_is_always_cpu(policy_on, monkeypatch, requested):
    monkeypatch.setenv("OMNIVOICE_DEVICE", requested)
    assert device_caps.requested_device_override() == "cpu"
    caps = device_caps.refresh()
    assert caps.family == "cpu"
    assert caps.available_families == ("cpu",)
    assert device_caps.gpu_probe_skipped()


def test_loader_picks_cpu_when_a_gpu_is_visible(policy_on, monkeypatch):
    import services.model_manager as mm

    def _fail(*_a, **_k):
        raise AssertionError("the CPU-only loader touched a GPU runtime")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=_fail, mem_get_info=_fail),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        get_num_threads=lambda: 1, set_num_threads=lambda n: None,
        get_num_interop_threads=lambda: 1, set_num_interop_threads=lambda n: None,
    )
    monkeypatch.setattr(mm, "_torch", fake_torch)
    monkeypatch.setenv("OMNIVOICE_FORCE_CUDA", "1")
    assert mm.get_best_device() == "cpu"
    mm.free_vram()  # must not reach torch.cuda.empty_cache
    mm.release_device_cache()
    assert mm._pick_gpu_workers() == 1


def test_environment_hides_gpus_from_child_processes(policy_on):
    env = {"OMNIVOICE_FORCE_CUDA": "1", "CUDA_VISIBLE_DEVICES": "0", "OMNIVOICE_DEVICE": "cuda"}
    cpu_only.apply_environment(env)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["HIP_VISIBLE_DEVICES"] == ""
    assert env["OMNIVOICE_DEVICE"] == "cpu"
    assert env["OMNIVOICE_TORCH_VARIANT"] == "cpu"
    assert env["OMNIVOICE_VOXCPM_DEVICE"] == "cpu"
    assert env["OMNIVOICE_INDEXTTS_FP16"] == "0"
    assert env["OMNIVOICE_DOTS_TTS_PRECISION"] == "float32"
    assert "OMNIVOICE_FORCE_CUDA" not in env
    intra, _inter = cpu_only.thread_plan()
    assert env["OMP_NUM_THREADS"] == str(intra)


def test_low_resource_mode_defaults_on_and_user_values_win(policy_on):
    assert cpu_only.low_resource_mode() is True
    env = {"OMNIVOICE_IDLE_TIMEOUT": "900"}
    cpu_only.apply_environment(env)
    assert env["OMNIVOICE_DUB_BATCH_WIDTH"] == "1"
    assert env["OMNIVOICE_NLLB_BATCH_SIZE"] == "1"
    assert env["OMNIVOICE_PRELOAD_TTS_ASR"] == "0"
    assert env["OMNIVOICE_FREE_VRAM_BEFORE_GENERATE"] == "always"
    assert env["OMNIVOICE_IDLE_TIMEOUT"] == "900"  # explicit value kept


def test_low_resource_mode_can_be_turned_off(policy_on, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_LOW_RESOURCE_MODE", "0")
    env: dict = {}
    cpu_only.apply_environment(env)
    assert "OMNIVOICE_DUB_BATCH_WIDTH" not in env
    assert env["OMNIVOICE_DEVICE"] == "cpu"  # the device pin is not optional


def test_thread_plan_follows_physical_cores(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_CPU_THREADS", raising=False)
    monkeypatch.setattr(cpu_only, "physical_cores", lambda: 2)
    assert cpu_only.thread_plan() == (2, 1)
    monkeypatch.setattr(cpu_only, "physical_cores", lambda: 4)
    assert cpu_only.thread_plan() == (4, 1)
    monkeypatch.setattr(cpu_only, "physical_cores", lambda: 8)
    assert cpu_only.thread_plan() == (8, 2)
    monkeypatch.setenv("OMNIVOICE_CPU_THREADS", "3")
    assert cpu_only.thread_plan() == (3, 1)


def test_physical_cores_respects_affinity(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 32)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1}, raising=False)
    assert cpu_only.physical_cores() == 2


def test_precision_and_device_are_cpu_float32():
    torch = pytest.importorskip("torch")
    from omnivoice.utils.dtype import tts_dtype_name

    assert cpu_only.resolve_device("cuda") == "cpu"
    assert cpu_only.device() == torch.device("cpu")
    assert cpu_only.dtype_name() == "float32"
    assert tts_dtype_name(cpu_only.resolve_device()) == "float32"


def test_dynamic_quantization_shrinks_linear_layers():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))
    sample = torch.randn(2, 64)
    reference = model(sample)
    quantized = cpu_only.quantize_dynamic(model)
    assert "Quantized" in type(quantized[0]).__name__ or "Dynamic" in repr(quantized[0])
    assert torch.allclose(quantized(sample), reference, atol=0.2)


def test_torch_load_memory_maps_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "weights.pt"
    torch.save({"w": torch.arange(1024, dtype=torch.float32)}, path)
    loaded = cpu_only.torch_load(path)
    assert loaded["w"].device.type == "cpu"
    assert loaded["w"].dtype == torch.float32
    assert int(loaded["w"][-1]) == 1023


def test_release_memory_reports_ram():
    snapshot = cpu_only.release_memory()
    assert snapshot["rss_mb"] > 0
    assert snapshot["peak_rss_mb"] >= 0


def test_default_install_is_cpu_only():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'default-groups = ["dev", "cpu"]' in pyproject
    requirements = (ROOT / "requirements-cpu.txt").read_text(encoding="utf-8")
    names = {
        line.split("==")[0].strip().lower()
        for line in requirements.splitlines()
        if "==" in line and not line.startswith((" ", "#"))
    }
    gpu_only = {n for n in names if n.startswith(("nvidia-cu", "nvidia-nv", "triton"))}
    gpu_only |= names & {"onnxruntime-gpu", "faiss-gpu", "bitsandbytes", "xformers", "cupy"}
    assert not gpu_only, f"GPU-only packages in the CPU requirements: {sorted(gpu_only)}"
    assert "torch==2.10.0+cpu" in requirements
