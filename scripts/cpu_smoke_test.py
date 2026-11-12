#!/usr/bin/env python3
"""CPU-only smoke test: one end-to-end voice generation, with peak RAM and time.

    uv run python scripts/cpu_smoke_test.py                 # KittenTTS (80 MB, ONNX)
    uv run python scripts/cpu_smoke_test.py --engine omnivoice --json out.json

Fails (exit 1) if any GPU runtime is reachable, the device does not resolve to
the CPU, or no audio comes out. The first run downloads the engine's weights
from Hugging Face; later runs work offline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from core import cpu_only  # noqa: E402

cpu_only.enforce()  # before torch / onnxruntime load

DEFAULT_TEXT = (
    "This is a CPU only smoke test. The quick brown fox jumps over the lazy dog, "
    "and the whole sentence is rendered without a graphics card."
)


class PeakRam:
    """Samples resident memory on a thread; the high-water mark is the result."""

    def __init__(self, interval_s: float = 0.05):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval_s
        self._stop = threading.Event()
        self.peak_mb = self._rss()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _rss(self) -> float:
        return self._proc.memory_info().rss / (1024 ** 2)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak_mb = max(self.peak_mb, self._rss())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, self._rss())


def assert_cpu_only() -> dict:
    import torch

    from services.model_manager import get_best_device

    cpu_only.configure_torch(torch)

    problems = []
    if torch.cuda.is_available():
        problems.append("torch.cuda.is_available() is True")
    device = get_best_device()
    if device != "cpu":
        problems.append(f"get_best_device() returned {device!r}")
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
    except ImportError:
        providers = []
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            problems.append("CTranslate2 sees a CUDA device")
    except ImportError:
        pass
    if problems:
        raise SystemExit("GPU reachable in a CPU-only build: " + "; ".join(problems))
    return {
        "device": device,
        "torch": torch.__version__,
        "torch_threads": [torch.get_num_threads(), torch.get_num_interop_threads()],
        "onnxruntime_providers": providers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", default="kittentts", help="TTS backend id (default: kittentts)")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--out", default="cpu_smoke.wav")
    parser.add_argument("--json", help="also write the measurements to this file")
    args = parser.parse_args()

    report: dict = {
        "engine": args.engine,
        "physical_cores": cpu_only.physical_cores(),
        "low_resource_mode": cpu_only.low_resource_mode(),
        "ram_before": cpu_only.ram_snapshot(),
    }
    with PeakRam() as ram:
        report.update(assert_cpu_only())

        from services.tts_backend import get_backend_class

        started = time.perf_counter()
        backend = get_backend_class(args.engine)()
        wav = backend.generate(args.text)
        report["load_and_generate_s"] = round(time.perf_counter() - started, 2)

        # Second pass: the model is warm, so this is inference time alone.
        started = time.perf_counter()
        wav = backend.generate(args.text)
        report["warm_generate_s"] = round(time.perf_counter() - started, 2)

        sample_rate = int(backend.sample_rate)
        samples = wav.detach().to("cpu").float().numpy().reshape(-1)
        if samples.size == 0 or not float(abs(samples).max()) > 0:
            raise SystemExit("the engine returned no audio")
        import soundfile

        soundfile.write(args.out, samples, sample_rate)
        report["audio_s"] = round(samples.size / sample_rate, 2)
        report["realtime_factor"] = round(report["warm_generate_s"] / report["audio_s"], 3)

        del backend, wav
        report["ram_after_unload"] = cpu_only.release_memory()
    report["peak_rss_mb"] = round(ram.peak_mb, 1)

    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"OK: {report['audio_s']}s of audio on {report['device']} in "
        f"{report['warm_generate_s']}s (RTF {report['realtime_factor']}), "
        f"peak RAM {report['peak_rss_mb']} MB -> {os.path.abspath(args.out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
