#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install with: pip install uv"
  exit 1
fi

host_arch="$(uname -m)"
preferred_python=""

if [[ -x /opt/homebrew/bin/python3.11 ]]; then
  py_arch="$(/opt/homebrew/bin/python3.11 -c 'import platform; print(platform.machine())')"
  if [[ "$py_arch" == "arm64" ]]; then
    preferred_python="/opt/homebrew/bin/python3.11"
  fi
fi

if [[ -z "$preferred_python" && -x /opt/homebrew/bin/python3.13 ]]; then
  py_arch="$(/opt/homebrew/bin/python3.13 -c 'import platform; print(platform.machine())')"
  if [[ "$py_arch" == "arm64" ]]; then
    preferred_python="/opt/homebrew/bin/python3.13"
  fi
fi

if [[ ! -d .venv ]]; then
  if [[ -n "$preferred_python" ]]; then
    uv venv --python "$preferred_python"
  elif command -v python3.10 >/dev/null 2>&1; then
    uv venv --python python3.10
  else
    uv venv
  fi
else
  if [[ "$host_arch" == "arm64" ]]; then
    venv_arch="$(.venv/bin/python -c 'import platform; print(platform.machine())')"
    if [[ "$venv_arch" == "x86_64" ]]; then
      echo "Recreating .venv with arm64 Python (current venv is x86_64)."
      rm -rf .venv
      if [[ -n "$preferred_python" ]]; then
        uv venv --python "$preferred_python"
      else
        uv venv
      fi
    fi
  fi
fi

uv pip install -r requirements-cpu.txt

uv run python download_reference_voices.py \
  --output-dir reference \
  --voices-file voices.json \
  --reset

mkdir -p models/VoxCPM1.5 models/onnx_models models/onnx_models_quantized

if ! ls models/VoxCPM1.5/*.safetensors >/dev/null 2>&1 && ! ls models/VoxCPM1.5/*.bin >/dev/null 2>&1; then
  echo "Downloading VoxCPM1.5 weights to models/VoxCPM1.5..."
  uv pip install -r requirements-export.txt
  uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="openbmb/VoxCPM1.5",
    local_dir="./models/VoxCPM1.5",
    local_dir_use_symlinks=False,
)
PY
fi

required_models=(
  VoxCPM_Text_Embed.onnx
  VoxCPM_VAE_Encoder.onnx
  VoxCPM_Feat_Encoder.onnx
  VoxCPM_Feat_Cond.onnx
  VoxCPM_Concat.onnx
  VoxCPM_Main.onnx
  VoxCPM_Feat_Decoder.onnx
  VoxCPM_VAE_Decoder.onnx
)

missing=()
for name in "${required_models[@]}"; do
  if [[ ! -f "models/onnx_models_quantized/${name}" ]]; then
    missing+=("$name")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing ONNX files in models/onnx_models_quantized. Exporting and quantizing..."
  uv pip install -r requirements-export.txt
  uv run python Export_VoxCPM_ONNX.py \
    --voxcpm-dir ./models/VoxCPM1.5 \
    --onnx-dir ./models/onnx_models

  uv run python Optimize_ONNX.py \
    --input-dir ./models/onnx_models \
    --output-dir ./models/onnx_models_quantized \
    --cpu
fi

if [[ ! -f "models/VoxCPM1.5/tokenizer.json" && ! -f "models/VoxCPM1.5/tokenizer.model" ]]; then
  echo "Tokenizer files not found in models/VoxCPM1.5."
  exit 1
fi

mkdir -p outputs

uv run python infer.py --config config.json

echo "Done. Output: outputs/demo.wav"
