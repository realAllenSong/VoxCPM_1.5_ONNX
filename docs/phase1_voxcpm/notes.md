# Notes: VoxCPM 1.5B ONNX CPU Inference

## Sources

### Source 1: Text-to-Speech-TTS-ONNX/VoxCPM (local)
- URL: https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX
- Key points:
  - `Inference_VoxCPM_ONNX.py` uses 8 ONNX parts (`VoxCPM_Text_Embed`, `VoxCPM_VAE_Encoder`, `VoxCPM_Feat_Encoder`, `VoxCPM_Feat_Cond`, `VoxCPM_Concat`, `VoxCPM_Main`, `VoxCPM_Feat_Decoder`, `VoxCPM_VAE_Decoder`).
  - Paths to VoxCPM model and ONNX files are hard-coded to `/home/DakeQQ/...`.
  - Voice cloning is supported via `prompt_audio_path` + `prompt_text`; if no prompt audio, it uses a random seed voice.
  - Uses `LlamaTokenizerFast.from_pretrained(path_voxcpm)` and `TextNormalizer`.
  - Sample rate fixed at 44.1kHz; `half_decode_len` fixed for VoxCPM1.5.
  - `Export_VoxCPM_ONNX.py` copies modified `model.py`, `core.py`, `audio_vae.py` into VoxCPM package/model path before exporting.
  - `Optimize_ONNX.py` has CPU quantization presets (int8 for key models) and expects a folder of ONNX models.

### Source 2: VoxCPM-ONNX (local)
- URL: https://github.com/bluryar/VoxCPM-ONNX
- Key points:
  - Works on CPU with a 0.5B ONNX export (4 ONNX files: prefill, decode, VAE enc/dec).
  - CLI `infer.py` exposes a clear argument interface (`--text`, `--models-dir`, `--prompt-audio`, `--prompt-text`, `--device`, `--dtype`).
  - Provides a lightweight NumPy+ONNXRuntime inference loop without PyTorch.

### Source 3: VoxCPM official (local)
- URL: https://github.com/OpenBMB/VoxCPM
- Key points:
  - Official weights: VoxCPM1.5 on Hugging Face/ModelScope.
  - Voice cloning is supported via prompt audio + prompt text; no explicit built-in voice list in repo (default is zero-shot prompt-free behavior).
  - Sample rate for 1.5 is 44.1kHz; patch-size=4; model params ~800M (per README).

### User requirements (2025-xx-xx)
- 1.5B model weights source: https://huggingface.co/openbmb/VoxCPM1.5
- Text normalizer must be enabled.
- Need both preset voice list and prompt-based voice cloning.
- Target GitHub Actions: Python 3.10 on standard runners.
- Project files should be placed at repo root; duplicates are acceptable.

## Synthesized Findings

### Current blockers
- No public 1.5B ONNX model found on Hugging Face (API search returned only `bluryar/voxcpm-onnx`).
- The 1.5B ONNX pipeline in `Text-to-Speech-TTS-ONNX/VoxCPM` is not parameterized and not packaged as a simple CLI.
- Export script mutates installed VoxCPM package files, which is fragile for automation.

### Model availability
- Hugging Face search for "VoxCPM ONNX" only returns the 0.5B ONNX repo.
- No 1.5B ONNX model found in GitHub search results.

### Proposed workflow
- Create a simple CLI wrapper for 1.5B ONNX inference with clear `--models-dir` and `--voxcpm-dir` inputs and optional voice cloning arguments.
- Provide optional download step for VoxCPM1.5 tokenizer/config from Hugging Face (for tokenizer) and a clear path for user-provided ONNX files.
- Add export + quantize scripts with CLI arguments and root-level defaults.

### Latest updates
- Added `scripts/download_reference_voices.py` to pull prompt audios from the VoxCPM demo page into `reference/` and update `voices.json`.
- Added `scripts/run_service.sh` to create a uv venv, install deps, sync voices, and run a mixed CN/EN test (requires models present).
- Export pipeline now uses local `VoxCPM/src` code to avoid heavy `voxcpm` pip deps; `requirements-export.txt` trimmed.
- `scripts/run_service.sh` successfully exported + quantized models and generated `outputs/demo.wav` on arm64 Python.
