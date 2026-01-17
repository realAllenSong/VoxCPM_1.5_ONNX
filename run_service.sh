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

VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Virtual environment not found at $VENV_PYTHON"
  exit 1
fi

VOXCPM_MODEL_REPO="${VOXCPM_MODEL_REPO:-openbmb/VoxCPM1.5}"
VOXCPM_ONNX_REPO="${VOXCPM_ONNX_REPO:-Oulasong/voxcpm-onnx}"
VOXCPM_ONNX_REVISION="${VOXCPM_ONNX_REVISION:-}"
VOXCPM_ONNX_URL="${VOXCPM_ONNX_URL:-}"
VOXCPM_ONNX_FORCE="${VOXCPM_ONNX_FORCE:-0}"
VOXCPM_USE_QUANTIZED="${VOXCPM_USE_QUANTIZED:-0}"

if [[ "$VOXCPM_USE_QUANTIZED" == "1" ]]; then
  ONNX_TARGET_DIR="models/onnx_models_quantized"
  echo "Using quantized ONNX models (may cause audio distortion on some Linux platforms)"
else
  ONNX_TARGET_DIR="models/onnx_models"
  echo "Using full-precision ONNX models (recommended for cross-platform compatibility)"
fi

uv pip install --python "$VENV_PYTHON" -r requirements-cpu.txt

uv run --python "$VENV_PYTHON" python download_reference_voices.py \
  --output-dir reference \
  --voices-file voices.json \
  --reset

if [[ ! -d "VoxCPM/src/voxcpm" ]]; then
  if [[ -d "VoxCPM" ]]; then
    backup_dir="VoxCPM.bak.$(date +%s)"
    echo "VoxCPM directory exists but VoxCPM/src/voxcpm is missing."
    echo "Backing up to ${backup_dir} and cloning a fresh copy."
    mv VoxCPM "$backup_dir"
  fi
  if command -v git >/dev/null 2>&1; then
    echo "VoxCPM source not found. Cloning OpenBMB/VoxCPM..."
    git clone --depth 1 https://github.com/OpenBMB/VoxCPM VoxCPM
  else
    echo "VoxCPM source not found and git is unavailable."
    echo "Please clone https://github.com/OpenBMB/VoxCPM into ./VoxCPM"
    exit 1
  fi
fi

mkdir -p models/VoxCPM1.5 models/onnx_models models/onnx_models_quantized

if ! ls models/VoxCPM1.5/*.safetensors >/dev/null 2>&1 && ! ls models/VoxCPM1.5/*.bin >/dev/null 2>&1; then
  echo "Downloading VoxCPM1.5 weights to models/VoxCPM1.5..."
  uv pip install --python "$VENV_PYTHON" -r requirements-export.txt
  uv run --python "$VENV_PYTHON" python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ.get("VOXCPM_MODEL_REPO", "openbmb/VoxCPM1.5"),
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

have_all_onnx() {
  local dir="$1"
  for name in "${required_models[@]}"; do
    if [[ ! -f "${dir}/${name}" ]]; then
      return 1
    fi
  done
  return 0
}

download_prebuilt_onnx() {
  local target_dir="$ONNX_TARGET_DIR"
  local download_dir="$target_dir"
  local tmp_dir=""

  if [[ -n "$(ls -A "$target_dir" 2>/dev/null)" ]]; then
    tmp_dir="$(mktemp -d)"
    download_dir="$tmp_dir"
  fi

  if [[ -n "$VOXCPM_ONNX_REPO" ]]; then
    echo "Attempting to download prebuilt ONNX from HF: ${VOXCPM_ONNX_REPO}"
    VOXCPM_ONNX_LOCAL_DIR="$download_dir" uv run --python "$VENV_PYTHON" python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ.get("VOXCPM_ONNX_REPO")
revision = os.environ.get("VOXCPM_ONNX_REVISION") or None
local_dir = os.environ.get("VOXCPM_ONNX_LOCAL_DIR")
allow_patterns = ["*.onnx", "*.onnx.data", "voxcpm_onnx_config.json"]
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
    allow_patterns=allow_patterns,
    local_dir_use_symlinks=False,
)
PY
  elif [[ -n "$VOXCPM_ONNX_URL" ]]; then
    echo "Attempting to download prebuilt ONNX from URL: ${VOXCPM_ONNX_URL}"
    archive_path="${download_dir}/onnx_models_quantized.archive"
    if command -v curl >/dev/null 2>&1; then
      curl -L "$VOXCPM_ONNX_URL" -o "$archive_path"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$archive_path" "$VOXCPM_ONNX_URL"
    else
      echo "Neither curl nor wget is available."
      return 1
    fi

    case "$VOXCPM_ONNX_URL" in
      *.tar.gz|*.tgz)
        tar -xzf "$archive_path" -C "$download_dir"
        ;;
      *.zip)
        unzip -q "$archive_path" -d "$download_dir"
        ;;
      *)
        echo "Unsupported archive format (use .tar.gz/.tgz/.zip)."
        rm -f "$archive_path"
        return 1
        ;;
    esac
    rm -f "$archive_path"
  else
    return 1
  fi

  if ! have_all_onnx "$download_dir"; then
    subdir="$(find "$download_dir" -mindepth 1 -maxdepth 1 -type d | head -n1 || true)"
    if [[ -n "$subdir" ]]; then
      shopt -s dotglob
      mv "$subdir"/* "$download_dir"/
      shopt -u dotglob
      rmdir "$subdir" 2>/dev/null || true
    fi
  fi

  if [[ -n "$tmp_dir" ]]; then
    if have_all_onnx "$tmp_dir"; then
      rm -rf "$target_dir"
      mv "$tmp_dir" "$target_dir"
    else
      echo "Prebuilt ONNX download incomplete. Keeping existing files."
      rm -rf "$tmp_dir"
      return 1
    fi
  fi

  return 0
}

if [[ "$VOXCPM_ONNX_FORCE" == "1" ]]; then
  download_prebuilt_onnx || true
fi

if ! have_all_onnx "$ONNX_TARGET_DIR"; then
  if [[ -n "$VOXCPM_ONNX_REPO" || -n "$VOXCPM_ONNX_URL" ]]; then
    download_prebuilt_onnx || true
  fi
fi

if ! have_all_onnx "$ONNX_TARGET_DIR"; then
  echo "Missing ONNX files in $ONNX_TARGET_DIR. Exporting..."
  uv pip install --python "$VENV_PYTHON" -r requirements-export.txt
  uv run --python "$VENV_PYTHON" python Export_VoxCPM_ONNX.py \
    --voxcpm-dir ./models/VoxCPM1.5 \
    --onnx-dir ./models/onnx_models

  if [[ "$VOXCPM_USE_QUANTIZED" == "1" ]]; then
    echo "Quantizing ONNX models for CPU..."
    uv run --python "$VENV_PYTHON" python Optimize_ONNX.py \
      --input-dir ./models/onnx_models \
      --output-dir ./models/onnx_models_quantized \
      --cpu
  fi
fi

if [[ ! -f "models/VoxCPM1.5/tokenizer.json" && ! -f "models/VoxCPM1.5/tokenizer.model" ]]; then
  echo "Tokenizer files not found in models/VoxCPM1.5."
  exit 1
fi

mkdir -p outputs

# Read model_size from config.json
MODEL_SIZE=$(uv run --python "$VENV_PYTHON" python -c "import json; print(json.load(open('config.json')).get('model_size', '1.5b').lower())" 2>/dev/null || echo "1.5b")

echo "Model size from config.json: $MODEL_SIZE"

if [[ "$MODEL_SIZE" == "0.5b" || "$MODEL_SIZE" == "05b" ]]; then
  echo "Running 0.5B model via engines abstraction..."
  
  # Check if 0.5B models exist
  if [[ ! -f "models/onnx_models_05b/voxcpm_prefill.onnx" ]]; then
    echo "0.5B ONNX models not found. Downloading from HuggingFace..."
    uv run --python "$VENV_PYTHON" python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='bluryar/voxcpm-onnx',
    local_dir='./models/onnx_models_05b',
    allow_patterns=['*.onnx', '*.onnx.data', 'tokenizer.json'],
)
print('Download complete!')
"
  fi
  
  # Run 0.5B test script
  uv run --python "$VENV_PYTHON" python test_05b.py
else
  echo "Running 1.5B model..."
  uv run --python "$VENV_PYTHON" python infer.py --config config.json
fi

echo "Done. Output: outputs/demo.wav"
