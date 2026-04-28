## Learned User Preferences

- When mirroring README setup, prefer **`uv`** (`uv venv`, `uv pip`) instead of **conda** for Python installs in this repo.

## Learned Workspace Facts

- On **macOS/Apple Silicon**, install **PyTorch from PyPI** (CPU/MPS). CUDA-specific wheel URLs from the README target **Linux + NVIDIA**, not Darwin wheels.
- **Triton** has no usable macOS arm64 wheels on PyPI; EDT and related SAM3 paths use **OpenCV CPU fallbacks** when Triton is absent.
- **`decord`** is often unavailable on macOS ARM; video loading falls back to **OpenCV** when `decord` cannot be imported.
- **`detectron2`** builds from source frequently fail on Apple toolchains; the repo supports a **Torchvision Faster R-CNN person detector** when Detectron2 is missing.
- **PyOpenGL/pyrender** on Darwin: EGL defaults often break **`pyrender`**; entrypoints set **`PYOPENGL_PLATFORM=pyglet`** early on macOS.
- **Diffusion-VAS on MPS**: prefer **`enable_model_cpu_offload(device=...)`** instead of `.to(mps)` on the full pipeline; load large checkpoints with **`map_location="cpu"`** before moving tensors to reduce peak unified memory use.
- For **OOM or SIGKILL** on Mac with MPS, lower **`sam_3d_body.batch_size`** (e.g. 8→4); optionally disable completion pipelines you do not need.
- SAM3 **video/tracker**: use **device-aware helpers** (`default_compute_device`, model `device`) instead of hardcoded **`cuda()`**/`torch.device("cuda")` for storage and tensors.
- **Weights**: set **`configs/body4d.yaml`** `paths.ckpt_root` (and related paths) to real checkpoint dirs; placeholders cause **`FileNotFoundError`** at runtime.
