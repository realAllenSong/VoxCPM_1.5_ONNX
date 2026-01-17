# VoxCPM 1.5B ONNX (CPU)

ONNX_Lab 致力于打造简单易用的强大开源 TTS 模型的 ONNX CPU 运行版，旨在以最小成本跑出最高质量语音。
目前支持 **VoxCPM 1.5B** （0.5B版本暂时不支持），后续会逐步扩展更多模型与推理方案。

这是一个放在仓库根目录的 VoxCPM 1.5B ONNX CPU 推理项目，基于 `Text-to-Speech-TTS-ONNX/VoxCPM` 改造，提供：

- 一键导出 ONNX
- CPU 量化优化
- 推理 CLI（默认启用 text-normalizer）
- 预置音色 + 语音克隆

## 目录结构 (建议)

```
.
├── Export_VoxCPM_ONNX.py
├── Optimize_ONNX.py
├── infer.py
├── config.json
├── voices.json
├── api_server.py                # FastAPI 服务（支持流式输出）
├── engines/                      # 引擎抽象层
│   ├── __init__.py
│   ├── base.py                   # 抽象基类
│   ├── voxcpm_15b.py             # 1.5B 引擎
│   └── voxcpm_05b.py             # 0.5B 引擎（实验性）
├── reference/                    # 官方示例音色 (prompt audio)
├── clone_reference/              # 用户自定义参考音色
├── download_reference_voices.py
├── run_service.sh
├── modeling_modified/
├── VoxCPM/                       # 官方源码（用于导出）
├── models/
│   ├── VoxCPM1.5/                # 官方权重 + tokenizer/config
│   ├── onnx_models/              # 导出的全精度 ONNX（推荐用于生产环境）
│   ├── onnx_models_05b/          # 0.5B ONNX（实验性）
│   └── onnx_models_quantized/    # CPU 量化后的 ONNX（实验性）
```


## 一键启动 (推荐)

```bash
./run_service.sh
```

这个脚本会自动：
- 下载 VoxCPM1.5 权重
- 如配置了预构建 ONNX，会优先下载并使用（否则走导出+量化）
- 导出 ONNX
- CPU 量化
- 同步官方示例音色
- 生成一段中英文混合测试音频到 `outputs/demo.wav`

### 🔥 性能优化提示

**预热模型** (减少首次请求延迟)：
```python
# 启动时预热
engine.synthesize(["warmup"], voice="default")  # 第一次调用慢，后续快
```

**流式播放** (边生成边播放，适合 LLM 场景)：
```python
# LLM 输出时按句拆分
for sentence in split_sentences(llm_output):
    for chunk in tts.stream(sentence):
        play_audio(chunk)  # 边生成边播放
```

**连接复用** (HTTP 请求)：
```python
session = requests.Session()  # 复用 TCP 连接
session.post("http://localhost:8000/synthesize", json=payload)
```

## ⚠️ 重要：模型版本选择

本项目提供两种 ONNX 模型版本：

### 1. 全精度版本 (`onnx_models/`) - **推荐**
- ✅ 跨平台兼容性好（Mac/Linux/Windows）
- ✅ 音质稳定可靠
- ⚠️ 体积较大（~6GB）
- ⚠️ 推理速度稍慢

### 2. 量化版本 (`onnx_models_quantized/`) - **实验性**
- ✅ 体积小（~1.5GB）
- ✅ 推理速度快
- ❌ **在某些 Linux 平台（Colab/GitHub Actions）上可能产生音频失真**
- ✅ Mac 上测试正常

**建议：**
- **生产环境、Colab、GitHub Actions**: 使用全精度版本
- **本地开发（Mac）**: 可尝试量化版本
- 如果遇到音频失真问题，请切换到全精度版本

## 预构建 ONNX 下载（加速 CI/Colab）

已在 Hugging Face 上传两个版本：
- 全精度：`Oulasong/voxcpm-onnx/onnx_models/` (main 分支)
- 量化：`Oulasong/voxcpm-onnx/onnx_models_quantized/` (quantized 分支)

通过环境变量指定下载版本：

```bash
# 使用全精度版本（推荐用于 Colab/GitHub Actions）
VOXCPM_USE_QUANTIZED=0 \
VOXCPM_ONNX_REPO=Oulasong/voxcpm-onnx \
VOXCPM_ONNX_FORCE=1 \
./run_service.sh

# 使用量化版本（仅推荐 Mac 本地使用）
VOXCPM_USE_QUANTIZED=1 \
VOXCPM_ONNX_REPO=Oulasong/voxcpm-onnx \
VOXCPM_ONNX_REVISION=quantized \
VOXCPM_ONNX_FORCE=1 \
./run_service.sh



说明：
- `VOXCPM_USE_QUANTIZED=0`: 使用全精度版本（默认，推荐）
- `VOXCPM_USE_QUANTIZED=1`: 使用量化版本（可能在某些平台失真）
- `VOXCPM_ONNX_FORCE=1`：即使本地已有缓存也会优先尝试下载；失败则保留已有文件。
- `VOXCPM_ONNX_REVISION`：指定 Hugging Face 分支（main=全精度，quantized=量化版）
- `VOXCPM_MODEL_REPO`：指定权重来源（默认 openbmb/VoxCPM1.5）



## 1) 使用 uv 管理环境

安装 uv：

```bash
pip install uv
```

创建虚拟环境并安装依赖：

```bash
uv venv
uv pip install -r requirements-cpu.txt
```

导出与量化需要额外依赖：

```bash
uv pip install -r requirements-export.txt
```

## 2) 下载官方示例音色 (reference)

```bash
uv run python download_reference_voices.py \
  --output-dir reference \
  --voices-file voices.json
```

如果需要重置为官方示例音色列表（保留 `default`），加上 `--reset`。

## 3) 下载 VoxCPM1.5 权重

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="openbmb/VoxCPM1.5",  # 可用 VOXCPM_MODEL_REPO 覆盖
    local_dir="./models/VoxCPM1.5",
    local_dir_use_symlinks=False,
)
```

## 4) 导出 ONNX

```bash
python Export_VoxCPM_ONNX.py \
  --voxcpm-dir ./models/VoxCPM1.5 \
  --onnx-dir ./models/onnx_models
```

默认不运行推理测试，如需导出后测试：

```bash
python Export_VoxCPM_ONNX.py \
  --voxcpm-dir ./models/VoxCPM1.5 \
  --onnx-dir ./models/onnx_models \
  --run-infer
```

## 5) CPU 量化优化

```bash
python Optimize_ONNX.py \
  --input-dir ./models/onnx_models \
  --output-dir ./models/onnx_models_quantized \
  --cpu
```

## 6) 推理

**直接合成** (默认读取 `./models/onnx_models_quantized` 和 `./models/VoxCPM1.5`):

```bash
python infer.py \
  --text "你好，我是 VoxCPM 1.5B 的 ONNX 版本。" \
  --output output.wav
```

**语音克隆** (prompt audio + prompt text):

```bash
python infer.py \
  --text "这是一个语音克隆测试。" \
  --prompt-audio ./clone_reference/prompt.wav \
  --prompt-text "参考语音的文字内容" \
  --output cloned.wav
```

**预置音色**:

```bash
python infer.py --list-voices
python infer.py --voice default --text "使用预置音色" --output preset.wav
```

修改 `voices.json` 可以添加你自己的音色。自定义参考音色可以放到 `clone_reference/`，然后用 `--prompt-audio/--prompt-text` 进行克隆。

## 在其他项目中调用（CLI）

在你的项目里生成临时 config 并直接调用本仓库的 `infer.py`：

```python
import json
import subprocess

cfg = {
  "models_dir": "/path/ONNX_Lab/models/onnx_models",  # 使用全精度版本（推荐）
  "voxcpm_dir": "/path/ONNX_Lab/models/VoxCPM1.5",
  "voices_file": "/path/your_project/voices.json",
  "voice": "default",
  "prompt_audio": None,
  "prompt_text": None,
  "text": "你好，这是测试。",
  "output": "out.wav"
}

with open("tmp_config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

subprocess.run(["python", "/path/ONNX_Lab/infer.py", "--config", "tmp_config.json"], check=True)
```

语音克隆时把 `voice` 设为 `null`，同时提供 `prompt_audio` + `prompt_text` 即可。
`voices.json` 中的相对路径会相对它自身所在目录解析。

## API 服务（FastAPI）

推荐：上传参考音频时用 **multipart**，仅使用预置音色时用 **JSON**。

> ⚡ **新增**：支持流式输出 (`/synthesize-stream`)，可实现低延迟实时 TTS 播放。

安装：

```bash
uv pip install -r requirements-api.txt
```

启动：

```bash
uv run python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

环境变量（可选）：

```bash
# 使用全精度版本（推荐）
export VOXCPM_MODELS_DIR=/path/ONNX_Lab/models/onnx_models
export VOXCPM_VOXCPM_DIR=/path/ONNX_Lab/models/VoxCPM1.5
export VOXCPM_VOICES_FILE=/path/your_project/voices.json
export VOXCPM_MAX_CONCURRENCY=1

# 或使用量化版本（仅 Mac，可能在 Linux 失真）
# export VOXCPM_MODELS_DIR=/path/ONNX_Lab/models/onnx_models_quantized
```

多用户建议：提高 `VOXCPM_MAX_CONCURRENCY` 或使用 `uvicorn --workers N`（每个 worker 会加载一份模型，占用更多内存）。

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/info` | GET | 获取音频参数 (sample_rate, bit_depth, channels) |
| `/voices` | GET | 列出可用音色 |
| `/synthesize` | POST | 合成语音，返回 WAV 文件 |
| `/synthesize-file` | POST | 支持上传参考音频的合成 |
| `/synthesize-stream` | POST | **流式输出** Raw PCM (int16)，适合实时播放 |

### JSON 请求示例（预置音色）

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，这是测试。","voice":"default"}' \
  --output out.wav
```

### 流式输出示例

```bash
# 获取音频参数
curl -s http://localhost:8000/info
# 返回: {"sample_rate": 44100, "bit_depth": 16, "channels": 1}

# 流式合成（返回 Raw PCM）
curl -X POST http://localhost:8000/synthesize-stream \
  -H "Content-Type: application/json" \
  -d '{"text":"这是流式输出测试"}' \
  --output stream.pcm

# 播放 PCM (Mac/Linux)
ffplay -f s16le -ar 44100 -ac 1 stream.pcm
```

响应头包含音频格式信息：
- `X-Sample-Rate`: 采样率 (44100)
- `X-Bit-Depth`: 位深 (16)
- `X-Channels`: 声道数 (1)

### multipart 请求示例（语音克隆）

```bash
curl -X POST http://localhost:8000/synthesize-file \
  -F "text=这是语音克隆测试。" \
  -F "prompt_text=参考语音的文字内容" \
  -F "prompt_audio=@/path/to/prompt.wav" \
  --output cloned.wav
```

### Python 客户端示例（流式播放）

```python
import requests
import pyaudio

# 获取音频参数
info = requests.get("http://localhost:8000/info").json()
print(f"Sample rate: {info['sample_rate']}Hz")

# 初始化 PyAudio
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=info['channels'],
    rate=info['sample_rate'],
    output=True
)

# 流式播放
with requests.post(
    "http://localhost:8000/synthesize-stream",
    json={"text": "实时流式播放测试"},
    stream=True
) as r:
    for chunk in r.iter_content(chunk_size=4096):
        stream.write(chunk)

stream.stop_stream()
stream.close()
p.terminate()
```

### 高级功能

#### 重试机制 (`retry_badcase`)

自动检测并重试生成异常的 "unstoppable" 情况：

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好世界",
    "voice": "default",
    "retry_badcase": true,
    "max_retries": 3,
    "length_ratio_threshold": 6.0
  }' \
  --output out.wav
```

#### Prompt Cache（多句连续生成）

支持 LLM 流式场景，保持多句生成的声音一致性：

```python
from engines.voxcpm_15b import VoxCPM15BEngine

engine = VoxCPM15BEngine(...)

# 构建 prompt cache
cache = engine.build_prompt_cache("prompt.wav", "参考文本")

# 多句流式生成
sentences = ["第一句。", "第二句。", "第三句。"]
for sentence in sentences:
    audio, sr = engine.synthesize([sentence], prompt_audio="prompt.wav", prompt_text="参考文本")
    # 播放或保存 audio
    # 可选：合并 cache 保持长期一致性
    # cache = engine.merge_prompt_cache(cache, generated_feats)
```


## 配置文件 (config.json)

`config.json` 是运行时配置（不是 ONNX 配置），必须是 **纯 JSON**。

### 字段说明

- `models_dir`: ONNX 模型目录路径。⚠️ **平台兼容性说明**：
  - **Colab/GitHub Actions/Linux**: 推荐使用 `models/onnx_models`（全精度版本）
  - **Mac 本地开发**: 可使用 `models/onnx_models_quantized`（量化版本，速度更快）
  - 量化版本在某些 Linux 平台可能产生音频失真，遇到问题请切换到全精度版本
  - 这里的 ONNX 来自 `openbmb/VoxCPM1.5` 权重导出，**并非官方直接提供的 ONNX**
  - 该目录包含 `voxcpm_onnx_config.json`（采样率、步数等默认值）
- `voxcpm_dir`: `openbmb/VoxCPM1.5` 权重目录（用于 tokenizer/config）。
- `voice`: 预置音色名称（来自 `voices.json`）。使用 `voice` 时请保持 `prompt_audio`/`prompt_text` 为 `null`。如果使用`prompt_audio`,则`voice`填null
- `prompt_audio`: 语音克隆参考音频路径（与 `prompt_text` 成对出现）。
  推荐 `wav`/`flac`/`ogg`，自动转单声道、重采样至 44.1k。
  默认最大 20 秒（超出会截断），推荐 3~15 秒干净语音。
- `prompt_text`: 参考音频的文字内容（必须与 `prompt_audio` 对应）。
- `text`: 待合成内容（字符串或数组）。数组表示多句输入，会在句子间插入短停顿。
  长文本可以直接写成一个字符串，但推荐拆句或使用 `text_file`。
- `text_file`: 文本文件路径（每行一句）。
- `output`: 输出 WAV 路径。
- `text_normalizer`: 文本归一化开关（默认 `true`），会处理数字/英文/标点等。
- `audio_normalizer`: 音频 RMS 归一化开关（默认 `false`），用于参考音频和输出音频的音量校准。
- `cfg_value`: CFG 引导尺度（越高越贴近提示语，但过高可能发闷/失真）。常用 2.0~3.0。
- `fixed_timesteps`: 解码步数（越高越慢，可能更稳）。常用 8~12。
- `seed`: 随机种子（用于可复现性）。
- `max_threads`: ONNXRuntime CPU 线程数（`0` = 自动；GitHub Actions 可设 2）。
- `model_size`: 模型大小选择（`"1.5b"` 或 `"0.5b"`）。⚠️ **0.5B 目前为实验性功能**。

> 语言支持：模型以中文/英文为主，其他语言不保证效果。

### 示例：预置音色

```json
{
  "voice": "default",
  "prompt_audio": null,
  "prompt_text": null
}
```

### 示例：自定义语音克隆

```json
{
  "voice": null,
  "prompt_audio": "clone_reference/my.wav",
  "prompt_text": "这是参考音频的文字内容。"
}
```

配置优先级：**命令行参数 > config.json > 默认值**。

## 官方音色说明

`download_reference_voices.py` 会从官方 demo page 下载 prompt 音频到 `reference/`，
并自动写入 `voices.json`（供 `--voice` 使用）。`reference/basic_ref_zh.wav` 为默认音色示例。
同时会下载官方 **Context-Aware Speech Generation** 示例音频并写入 `voices.json`
（例如 `context_zh_story_telling`）。

## Context-Aware Speech Generation (w/o prompt speech)

该模式 **不使用 prompt audio**，仅依赖文本本身的语义让模型自动推断语气与风格。
示例文本已整理在 `context_aware_texts.json`，其中 `type` 只是描述标签 **不作为输入**。

```json
{
  "voice": null,
  "prompt_audio": null,
  "prompt_text": null,
  "text": ["在很久很久以前，有一个国王。"]
}
```

直接命令行也可：

```bash
python infer.py --text "在很久很久以前，有一个国王。" --output out.wav
```

## Normalizer

默认开启 text-normalizer；如需关闭：

```bash
python infer.py --no-text-normalizer --text "..." --output out.wav
```

或在 `config.json` 中设置 `text_normalizer: false`。

音频归一化默认关闭；如需开启：

```bash
python infer.py --audio-normalizer --text "..." --output out.wav
```

或在 `config.json` 中设置 `audio_normalizer: true`。

## Colab 快速体验

⚠️ **Colab 必须使用全精度版本，量化版本会导致音频失真！**

```python
# 1. 克隆仓库并安装 uv
!git clone https://github.com/realAllenSong/ONNX_Lab.git
%cd ONNX_Lab
!pip install uv
!chmod +x run_service.sh

# 2. 下载全精度 ONNX 模型并生成音频（推荐）
import os
os.environ['VOXCPM_USE_QUANTIZED'] = '0'
os.environ['VOXCPM_ONNX_REPO'] = 'Oulasong/voxcpm-onnx'
os.environ['VOXCPM_ONNX_FORCE'] = '1'
!./run_service.sh

# 3. 播放输出音频
from IPython.display import Audio
Audio("outputs/demo.wav")
```

## GitHub Actions（下载优先 + cache fallback）

⚠️ **GitHub Actions 必须使用全精度版本，量化版本会导致音频失真！**

推荐流程：先尝试下载预构建全精度 ONNX，失败则使用 cache（或走本地导出）。
示例 workflow：`.github/workflows/voxcpm_cpu.yml`

```yaml
name: voxcpm-cpu
on:
  workflow_dispatch:

jobs:
  infer:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install uv
        run: pip install uv
      - name: Restore cache
        uses: actions/cache@v4
        with:
          path: |
            models/VoxCPM1.5
            models/onnx_models
          key: voxcpm-onnx-full-${{ runner.os }}-v2
      - name: Run
        env:
          VOXCPM_USE_QUANTIZED: "0"  # 使用全精度版本
          VOXCPM_ONNX_REPO: Oulasong/voxcpm-onnx
          VOXCPM_ONNX_FORCE: "1"
        run: |
          chmod +x run_service.sh
          ./run_service.sh
      - name: Upload output
        uses: actions/upload-artifact@v4
        with:
          name: demo-audio
          path: outputs/demo.wav
```

## Credits

- Text-to-Speech-TTS-ONNX: https://github.com/DakeQQ/Text-to-Speech-TTS-ONNX
- VoxCPM-ONNX: https://github.com/bluryar/VoxCPM-ONNX
- VoxCPM (official): https://github.com/OpenBMB/VoxCPM
