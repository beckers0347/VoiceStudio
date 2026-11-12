# CPU-only migration (`feat/cpu-lightweight`)

This branch runs VoiceStudio on the CPU and system RAM only. No GPU is probed,
no GPU wheels are installed, and no API or UI workflow changes.

## What changed

| Area | Change | File |
|---|---|---|
| Device manager | New single owner of device, precision, threads and RAM release | `backend/core/cpu_only.py` |
| Device choke point | The compute-device override always answers `cpu`, so the probe never touches a GPU runtime | `backend/core/device_caps.py` |
| Startup | GPUs hidden and thread pools sized before torch / CTranslate2 / ONNX Runtime load | `backend/main.py` |
| Loader | Torch threads set at import; cache flush replaced by `gc` + `malloc_trim`; opt-in int8 quantization | `backend/services/model_manager.py` |
| Sidecar engines | Inherit `CUDA_VISIBLE_DEVICES=""`, `OMNIVOICE_DEVICE=cpu`, float32 switches; two engines that pick an accelerator themselves now honour the pin | `backend/engines/confucius4/main.py`, `backend/engines/moss_tts_v15/main.py` |
| Long text | Default TTS chunk is 400 characters (was 800) in low-resource mode | `backend/services/chunked_tts.py` |
| Desktop installer | Always installs the CPU PyTorch build and pins the backend to CPU | `electron/src/main/index.ts` |
| Dependencies | `uv sync` defaults to the `cpu` group; pip users get `requirements-cpu.txt` | `pyproject.toml`, `requirements-cpu.txt` |
| Tests | Policy tests; CUDA-group test names its group explicitly; legacy GPU tests run with a pytest-only escape | `tests/test_cpu_only_policy.py`, `tests/test_cpu_install.py`, `conftest.py` |
| Smoke test | End-to-end generation with peak RAM and timing | `scripts/cpu_smoke_test.py` |

## How the policy works

- **Device.** `core.cpu_only.enabled()` is true in every running app. The
  existing choke point (`requested_device_override`) returns `cpu`, so routing,
  `/health`, the loader and every engine badge inherit it.
- **Whole process tree.** `apply_environment()` overwrites
  `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
  `OMNIVOICE_DEVICE`, `OMNIVOICE_TORCH_VARIANT`, and removes
  `OMNIVOICE_FORCE_CUDA`. Sidecar engines run in their own virtualenvs and
  probe `torch.cuda` themselves; they inherit this environment.
- **Precision.** float32 for the in-process model (already the CPU policy in
  `omnivoice/utils/dtype.py`); IndexTTS fp16 and dots-tts bf16 are switched off.
  Whisper on CTranslate2 already uses int8 on CPU.
- **Threads.** Intra-op threads equal physical cores (capped by the affinity
  mask); inter-op is 1 up to four cores, 2 above. `OMNIVOICE_CPU_THREADS`
  overrides. OpenMP / MKL / OpenBLAS get the same count.
- **Memory.** `release_memory()` runs `gc.collect()` and `malloc_trim(0)`;
  `ram_snapshot()` reports RSS, peak RSS and available RAM via `psutil`.

## `low_resource_mode`

On by default. Turn off with `OMNIVOICE_LOW_RESOURCE_MODE=0` or
`"low_resource_mode": false` in `prefs.json`. The device pin is not affected.
Each item is a default; an explicit value for the same variable still wins.

| Setting | Value | Effect |
|---|---|---|
| `OMNIVOICE_DUB_BATCH_WIDTH`, `OMNIVOICE_NLLB_BATCH_SIZE` | 1 | Batch size 1 |
| `OMNIVOICE_GPU_WORKERS` | 1 | One model call at a time |
| `OMNIVOICE_CPU_POOL` | min(2, cores) | Fewer background workers |
| `OMNIVOICE_LLM_CONCURRENCY` | 2 | Was 6 |
| `OMNIVOICE_PRELOAD_TTS_ASR`, `_CAPTURE_ASR`, `_WATERMARK` | 0 | Nothing loads until needed |
| `OMNIVOICE_FREE_VRAM_BEFORE_GENERATE` | always | Idle ASR and side caches are dropped before every generate |
| `OMNIVOICE_IDLE_TIMEOUT` | 120 s | Idle TTS model unloads after 2 minutes (was 15) |
| TTS chunk size | 400 chars | Peak RAM follows the chunk, not the script length |

## Install

```
uv sync                                  # CPU wheels by default on this branch
pip install -r requirements-cpu.txt      # pip alternative
uv run python scripts/cpu_smoke_test.py
```

## Verification

- `tests/test_cpu_only_policy.py`: 17 tests. They cover a host where
  `torch.cuda.is_available()` is true and `OMNIVOICE_FORCE_CUDA=1` is set (the
  loader still returns `cpu` and never calls a GPU API), the child-process
  environment, thread sizing, quantization, mmap loading, and that
  `requirements-cpu.txt` holds no `nvidia-cu*`, `triton`, `onnxruntime-gpu`,
  `faiss-gpu`, `bitsandbytes` or `xformers`.
- Full suites (`tests`, `backend/tests`): 15,846 passed before the final fixes.
  Three failures were caused by this branch and are fixed. The remaining 21
  fail identically on unmodified `main` in the same sandbox (macOS
  notarization, URL-import network guards, packaging layout and similar).
- Installed environment: `torch 2.10.0+cpu`, no `nvidia_*` packages, 2.3 GB
  virtualenv.

### Smoke test result

4 physical cores, 16 GB RAM, Linux x86_64, KittenTTS (ONNX Runtime, CPU provider):

| Measure | Value |
|---|---|
| Device / torch | `cpu` / `2.10.0+cpu`, threads 4 intra / 1 inter |
| Audio produced | 13.28 s |
| Cold load + generate | 14.3 s |
| Warm generate | 10.1–11.6 s (0.76–0.88× realtime) |
| Peak RAM | 843–869 MB |
| RAM after unload | 415 MB |

No GPU baseline exists for comparison: this sandbox has no GPU, and the
2.4 GB OmniVoice model was not benchmarked here.

## Known limits

- **GPU code is dormant, not deleted.** A grep for `cuda` still finds it. About
  300 existing tests exercise it and deleting it would stop the branch merging
  `main`. It is unreachable at runtime: the wheels have no CUDA, the devices
  are hidden, and the choke point is pinned.
- **Test escape.** `OMNIVOICE_TEST_ALLOW_GPU_PATHS=1` disables the policy, only
  when pytest is running.
- **Quantization is opt-in** (`OMNIVOICE_CPU_QUANTIZE=1`). It is verified on a
  small model only; its effect on OmniVoice audio quality is unmeasured.
  `torch.ao.quantization` is deprecated upstream.
- **GPU-only engines** (Ming TTS, MOSS SFX) have no CPU path and stay
  unavailable. CosyVoice 3 (9.8 GB) and VoxCPM2 (5 GB) run but are impractical
  on low-RAM machines; prefer KittenTTS, Supertonic, PocketTTS or the GGUF
  engines.
- **Sample rate is unchanged.** Engines emit their native rate; resampling
  down would add work and lose quality for no RAM saving.
- **mmap.** Model weights load through safetensors, which is already
  memory-mapped. `cpu_only.torch_load` is provided for `.pt` checkpoints; the
  only runtime `torch.load` is a kilobyte-sized voice-prompt cache, left as is.
- **Not done:** no CPU Docker image (`deploy/Dockerfile` is still CUDA-based,
  though the backend inside it would run on CPU), no Settings toggle for
  `low_resource_mode`, and the Electron change is not covered by a test.
- macOS is covered by the device pin only; PyPI torch there still contains MPS
  code, unused.
