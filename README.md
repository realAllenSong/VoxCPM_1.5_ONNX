# VoxCPM 1.5B ONNX (CPU)

ONNX_Lab 致力于打造简单易用的强大开源 TTS 模型的 ONNX CPU 运行版，旨在以最小成本跑出最高质量语音。
目前支持 **VoxCPM 1.5B**，后续会逐步扩展更多模型与推理方案。

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
├── reference/                   # 官方示例音色 (prompt audio)
├── clone_reference/             # 用户自定义参考音色
├── download_reference_voices.py
├── run_service.sh
├── modeling_modified/
├── VoxCPM/                      # 官方源码（用于导出）
├── models/
│   ├── VoxCPM1.5/                 # 官方权重 + tokenizer/config
│   ├── onnx_models/               # 导出的原始 ONNX
│   └── onnx_models_quantized/     # CPU 量化后的 ONNX
```


## 一键启动 (推荐)

```bash
./run_service.sh
```

这个脚本会自动：
- 下载 VoxCPM1.5 权重
- 导出 ONNX
- CPU 量化
- 同步官方示例音色
- 生成一段中英文混合测试音频到 `outputs/demo.wav`



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
    repo_id="openbmb/VoxCPM1.5",
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

## 配置文件 (config.json)

`config.json` 是运行时配置（不是 ONNX 配置），必须是 **纯 JSON**。

### 字段说明

- `models_dir`: ONNX 模型目录（**CPU 推荐用量化后的** `onnx_models_quantized`）。这里的 ONNX 来自
  `openbmb/VoxCPM1.5` 权重导出与量化，**并非官方直接提供的 ONNX**。
  该目录包含 `voxcpm_onnx_config.json`（采样率、步数等默认值）。
- `voxcpm_dir`: `openbmb/VoxCPM1.5` 权重目录（用于 tokenizer/config）。
- `voice`: 预置音色名称（来自 `voices.json`）。使用 `voice` 时请保持 `prompt_audio`/`prompt_text` 为 `null`。
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

> 将 `<your-org-or-user>` 替换为你的 GitHub 用户/组织名。

```python
# 1. 克隆仓库并安装 uv
!git clone https://github.com/realAllenSong/ONNX_Lab.git
%cd ONNX_Lab
!pip install uv
!chmod +x run_service.sh

# 2. 一键下载权重 + 导出 ONNX + 量化 + 生成音频
!./run_service.sh

# 3. 播放输出音频
from IPython.display import Audio
Audio("outputs/demo.wav")
```
