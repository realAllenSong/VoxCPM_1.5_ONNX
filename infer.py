#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import LlamaTokenizerFast

try:
    from modeling_modified.text_normalize import TextNormalizer
except Exception as exc:  # pragma: no cover - optional dependency
    TextNormalizer = None
    TEXT_NORMALIZER_IMPORT_ERROR = exc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "models", "onnx_models_quantized")
DEFAULT_VOXCPM_DIR = os.path.join(BASE_DIR, "models", "VoxCPM1.5")
DEFAULT_VOICES_FILE = os.path.join(BASE_DIR, "voices.json")
DEFAULT_RUN_CONFIG = os.path.join(BASE_DIR, "config.json")


DEFAULT_CONFIG = {
    "max_seq_len": 1024,
    "min_seq_len": 2,
    "decode_limit_factor": 6,
    "in_sample_rate": 44100,
    "out_sample_rate": 44100,
    "fixed_timesteps": 10,
    "cfg_value": 2.5,
    "random_seed": 1,
    "max_prompt_audio_seconds": 20,
    "half_decode_len": 7056,
    "blank_duration": 0.1,
}

REQUIRED_ONNX = [
    "VoxCPM_Text_Embed.onnx",
    "VoxCPM_VAE_Encoder.onnx",
    "VoxCPM_Feat_Encoder.onnx",
    "VoxCPM_Feat_Cond.onnx",
    "VoxCPM_Concat.onnx",
    "VoxCPM_Main.onnx",
    "VoxCPM_Feat_Decoder.onnx",
    "VoxCPM_VAE_Decoder.onnx",
]

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer.model",
]

STOP_TOKEN = [1]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoxCPM 1.5B ONNX CPU inference")
    parser.add_argument("--models-dir", default=None, help="Directory with VoxCPM ONNX files")
    parser.add_argument("--voxcpm-dir", default=None, help="VoxCPM1.5 model dir (for tokenizer/config)")
    parser.add_argument("--text", action="append", help="Text to synthesize (repeatable)")
    parser.add_argument("--text-file", default=None, help="Text file with one sentence per line")
    parser.add_argument("--output", default=None, help="Output WAV path")
    parser.add_argument("--prompt-audio", default=None, help="Reference audio path for voice cloning")
    parser.add_argument("--prompt-text", default=None, help="Reference text for the prompt audio")
    parser.add_argument("--prompt-text-file", default=None, help="Reference text file for the prompt audio")
    parser.add_argument("--voice", default=None, help="Use a preset voice name from voices.json")
    parser.add_argument("--voices-file", default=None, help="Voice preset JSON file")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    parser.add_argument("--config", default=None, help="Run config JSON (defaults to ./config.json if present)")
    parser.add_argument("--onnx-config", default=None, help="ONNX config JSON (defaults to models-dir/voxcpm_onnx_config.json)")
    parser.add_argument("--cfg-value", type=float, default=None, help="Override CFG value")
    parser.add_argument("--fixed-timesteps", type=int, default=None, help="Override fixed timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--max-threads", type=int, default=None, help="Max CPU threads (0 = auto)")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming decode")
    parser.add_argument("--no-text-normalizer", action="store_true", help="Disable text normalizer")
    parser.add_argument("--audio-normalizer", action="store_true", help="Enable audio normalizer")
    return parser.parse_args()


def load_run_config(config_path: str | None) -> tuple[dict, str | None]:
    if config_path is None:
        config_path = DEFAULT_RUN_CONFIG if os.path.isfile(DEFAULT_RUN_CONFIG) else None
    if config_path is None:
        return {}, None
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Run config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError("Run config must be a JSON object")
    return loaded, config_path


def load_onnx_config(models_dir: str, config_path: str | None) -> tuple[dict, str | None]:
    config = dict(DEFAULT_CONFIG)
    if config_path is None:
        config_path = os.path.join(models_dir, "voxcpm_onnx_config.json")
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            config.update(loaded)
    return config, config_path


def resample_audio_linear(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    duration = audio.shape[0] / float(sr_in)
    new_len = max(int(round(duration * sr_out)), 1)
    x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def read_audio_mono_int16(path: str, target_sr: int, max_samples: int | None) -> np.ndarray:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.dtype.kind in ("i", "u"):
        max_val = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = resample_audio_linear(audio, sr, target_sr)
    if max_samples is not None and audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    audio = np.clip(audio * 32768.0, -32768.0, 32767.0).astype(np.int16)
    return audio


def audio_normalizer(audio: np.ndarray, target_value: float = 8192.0) -> np.ndarray:
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean((audio * audio), dtype=np.float32), dtype=np.float32)
    audio *= (target_value / (rms + 1e-7))
    np.clip(audio, -32768.0, 32767.0, out=audio)
    return audio.astype(np.int16)


def mask_multichar_chinese_tokens(tokenizer):
    vocab = tokenizer.vocab if hasattr(tokenizer, "vocab") else tokenizer.get_vocab()
    multichar_tokens = {
        token for token in vocab.keys()
        if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        def __init__(self, base_tokenizer) -> None:
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs):
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")
            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []
            for token in tokens:
                clean_token = token.replace("▁", "")
                if clean_token in self.multichar_tokens:
                    processed.extend(list(clean_token))
                else:
                    processed.append(token)
            return processed

        def __call__(self, text: str, **kwargs):
            try:
                tokens = self.tokenize(text, **kwargs)
                return self.tokenizer.convert_tokens_to_ids(tokens)
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)


def load_texts(args: argparse.Namespace, run_config: dict) -> list[str]:
    texts: list[str] = []
    if args.text:
        for item in args.text:
            if item and item.strip():
                texts.append(item.strip())
    text_file = args.text_file
    if not texts and not text_file:
        cfg_text = run_config.get("text")
        if isinstance(cfg_text, str) and cfg_text.strip():
            texts.append(cfg_text.strip())
        elif isinstance(cfg_text, list):
            for item in cfg_text:
                if isinstance(item, str) and item.strip():
                    texts.append(item.strip())
        cfg_text_file = run_config.get("text_file")
        if not texts and isinstance(cfg_text_file, str) and cfg_text_file.strip():
            text_file = cfg_text_file.strip()

    if text_file:
        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    if not texts:
        raise ValueError("No text provided. Use --text or --text-file.")
    return texts


def apply_run_config(args: argparse.Namespace, run_config: dict, provided_flags: set[str]) -> None:
    def set_if_none(attr: str, key: str) -> None:
        if getattr(args, attr) is None and key in run_config:
            setattr(args, attr, run_config[key])

    set_if_none("models_dir", "models_dir")
    set_if_none("voxcpm_dir", "voxcpm_dir")
    set_if_none("output", "output")
    set_if_none("voice", "voice")
    set_if_none("prompt_audio", "prompt_audio")
    set_if_none("prompt_text", "prompt_text")
    set_if_none("prompt_text_file", "prompt_text_file")
    set_if_none("voices_file", "voices_file")
    set_if_none("text_file", "text_file")
    set_if_none("cfg_value", "cfg_value")
    set_if_none("fixed_timesteps", "fixed_timesteps")
    set_if_none("seed", "seed")
    set_if_none("max_threads", "max_threads")

    if "streaming" in run_config and "--streaming" not in provided_flags:
        args.streaming = bool(run_config["streaming"])

    if "audio_normalizer" in run_config and "--audio-normalizer" not in provided_flags:
        args.audio_normalizer = bool(run_config["audio_normalizer"])

    if "text_normalizer" in run_config and "--no-text-normalizer" not in provided_flags:
        args.no_text_normalizer = not bool(run_config["text_normalizer"])


def ensure_paths(models_dir: str, voxcpm_dir: str, prompt_audio: str | None) -> None:
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not os.path.isdir(voxcpm_dir):
        raise FileNotFoundError(f"VoxCPM directory not found: {voxcpm_dir}")

    missing = [name for name in REQUIRED_ONNX if not os.path.isfile(os.path.join(models_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing ONNX files in {models_dir}: {', '.join(missing)}")

    if not any(os.path.isfile(os.path.join(voxcpm_dir, name)) for name in TOKENIZER_FILES):
        raise FileNotFoundError(
            f"Tokenizer files not found in {voxcpm_dir}. Expected one of: {', '.join(TOKENIZER_FILES)}"
        )

    if prompt_audio and not os.path.isfile(prompt_audio):
        raise FileNotFoundError(f"Prompt audio not found: {prompt_audio}")


def load_voice_presets(voices_file: str) -> dict:
    if not os.path.isfile(voices_file):
        raise FileNotFoundError(f"Voices file not found: {voices_file}")
    with open(voices_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("voices.json must be a JSON object mapping voice names to settings")
    return data


def resolve_voice_prompt(voices_file: str, voice: str) -> tuple[str, str]:
    voices = load_voice_presets(voices_file)
    if voice not in voices:
        raise KeyError(f"Voice preset '{voice}' not found in {voices_file}")
    entry = voices[voice]
    if not isinstance(entry, dict):
        raise ValueError(f"Voice preset '{voice}' must be a JSON object")
    prompt_audio = entry.get("prompt_audio")
    prompt_text = entry.get("prompt_text")
    if not prompt_audio or not prompt_text:
        raise ValueError(f"Voice preset '{voice}' must include prompt_audio and prompt_text")
    if not os.path.isabs(prompt_audio):
        base_dir = os.path.dirname(os.path.abspath(voices_file))
        prompt_audio = os.path.normpath(os.path.join(base_dir, prompt_audio))
    return prompt_audio, prompt_text


def main() -> None:
    args = parse_args()
    provided_flags = {arg.split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")}
    run_config, run_config_path = load_run_config(args.config)
    apply_run_config(args, run_config, provided_flags)

    if args.models_dir is None:
        args.models_dir = DEFAULT_MODELS_DIR
    if args.voxcpm_dir is None:
        args.voxcpm_dir = DEFAULT_VOXCPM_DIR
    if args.output is None:
        args.output = "output.wav"
    if args.voices_file is None:
        args.voices_file = DEFAULT_VOICES_FILE
    if args.max_threads is None:
        args.max_threads = 0

    if args.list_voices:
        voices = load_voice_presets(args.voices_file)
        print("Available voices:")
        for name in sorted(voices.keys()):
            print(f"- {name}")
        return

    texts = load_texts(args, run_config)

    prompt_text = args.prompt_text
    if args.prompt_text_file:
        if prompt_text:
            raise ValueError("Provide only one of --prompt-text or --prompt-text-file.")
        with open(args.prompt_text_file, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()

    if args.voice:
        if args.prompt_audio or prompt_text:
            raise ValueError("Use either --voice or --prompt-audio/--prompt-text, not both.")
        prompt_audio, prompt_text = resolve_voice_prompt(args.voices_file, args.voice)
        args.prompt_audio = prompt_audio
        print(f"Using voice preset: {args.voice}")

    if (args.prompt_audio is None) != (prompt_text is None):
        raise ValueError("prompt audio and prompt text must both be provided or both be omitted")

    ensure_paths(args.models_dir, args.voxcpm_dir, args.prompt_audio)

    config, config_path = load_onnx_config(args.models_dir, args.onnx_config)

    max_seq_len = int(config["max_seq_len"])
    min_seq_len = int(config["min_seq_len"])
    decode_limit_factor = int(config["decode_limit_factor"])
    in_sample_rate = int(config["in_sample_rate"])
    out_sample_rate = int(config["out_sample_rate"])
    fixed_timesteps = int(args.fixed_timesteps if args.fixed_timesteps is not None else config["fixed_timesteps"])
    cfg_value = float(args.cfg_value if args.cfg_value is not None else config["cfg_value"])
    random_seed = int(args.seed if args.seed is not None else config["random_seed"])
    max_prompt_seconds = int(config.get("max_prompt_audio_seconds", 20))
    max_prompt_audio_len = max_prompt_seconds * in_sample_rate
    half_decode_len = int(config.get("half_decode_len", 7056))
    blank_duration = float(config.get("blank_duration", 0.1))

    use_text_normalizer = not args.no_text_normalizer
    text_normalizer = None
    if use_text_normalizer:
        if TextNormalizer is None:
            raise RuntimeError(
                "TextNormalizer unavailable. Install dependencies: pip install wetext regex inflect "
                f"(error: {TEXT_NORMALIZER_IMPORT_ERROR})"
            )
        else:
            text_normalizer = TextNormalizer()

    use_audio_normalizer = args.audio_normalizer

    if config_path and os.path.isfile(config_path):
        print(f"Loaded config: {config_path}")
    if run_config_path:
        print(f"Loaded run config: {run_config_path}")

    ort.set_seed(random_seed)
    session_opts = ort.SessionOptions()
    run_options = ort.RunOptions()

    session_opts.log_severity_level = 4
    session_opts.log_verbosity_level = 4
    run_options.log_severity_level = 4
    run_options.log_verbosity_level = 4

    session_opts.inter_op_num_threads = args.max_threads
    session_opts.intra_op_num_threads = args.max_threads
    session_opts.enable_cpu_mem_arena = True
    session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
    session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
    session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    session_opts.add_session_config_entry("session.graph_optimizations_loop_level", "2")
    session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
    session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
    session_opts.add_session_config_entry("optimization.enable_cast_chain_elimination", "1")
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

    providers = ["CPUExecutionProvider"]
    provider_options = None
    device_type = "cpu"
    device_id = 0

    def model_path(name: str) -> str:
        return os.path.join(args.models_dir, name)

    ort_session_A = ort.InferenceSession(model_path("VoxCPM_Text_Embed.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
    in_name_A = ort_session_A.get_inputs()[0].name
    out_name_A = [ort_session_A.get_outputs()[0].name]

    ort_session_B = ort.InferenceSession(model_path("VoxCPM_VAE_Encoder.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
    in_name_B = ort_session_B.get_inputs()[0].name
    out_name_B = [ort_session_B.get_outputs()[0].name]

    ort_session_C = ort.InferenceSession(model_path("VoxCPM_Feat_Encoder.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
    in_name_C = ort_session_C.get_inputs()[0].name
    out_name_C = [ort_session_C.get_outputs()[0].name]

    ort_session_D = ort.InferenceSession(model_path("VoxCPM_Feat_Cond.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
    model_dtype_D = ort_session_D._inputs_meta[0].type
    model_dtype_D = np.float16 if "float16" in model_dtype_D else np.float32
    in_name_D = ort_session_D.get_inputs()[0].name
    out_name_D = [ort_session_D.get_outputs()[0].name]

    ort_session_E = ort.InferenceSession(model_path("VoxCPM_Concat.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
    in_name_E = [item.name for item in ort_session_E.get_inputs()]
    out_name_E = [item.name for item in ort_session_E.get_outputs()]

    ort_session_F = ort.InferenceSession(model_path("VoxCPM_Main.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
    print(f"Usable Providers: {ort_session_F.get_providers()}")
    model_dtype_F = ort_session_F._inputs_meta[0].type
    model_dtype_F = np.float16 if "float16" in model_dtype_F else np.float32
    in_name_F = [item.name for item in ort_session_F.get_inputs()]
    out_name_F = [item.name for item in ort_session_F.get_outputs()]
    amount_of_outputs_F = len(out_name_F)

    ort_session_G = ort.InferenceSession(model_path("VoxCPM_Feat_Decoder.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
    model_dtype_G = ort_session_G._inputs_meta[2].type
    model_dtype_G = np.float16 if "float16" in model_dtype_G else np.float32
    in_name_G = [item.name for item in ort_session_G.get_inputs()]
    out_name_G = [item.name for item in ort_session_G.get_outputs()]

    ort_session_H = ort.InferenceSession(model_path("VoxCPM_VAE_Decoder.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
    model_dtype_H = ort_session_H._inputs_meta[0].type
    model_dtype_H = np.float16 if "float16" in model_dtype_H else np.float32

    shape_value_in_H = ort_session_H._inputs_meta[0].shape[1]
    dynamic_shape_vae_decode = isinstance(shape_value_in_H, str)

    in_name_H = ort_session_H.get_inputs()[0].name
    out_name_H = [item.name for item in ort_session_H.get_outputs()]

    generate_limit = max_seq_len - 1
    num_keys_values = amount_of_outputs_F - 4
    num_layers = num_keys_values // 2

    num_keys_values_plus_1 = num_keys_values + 1
    num_keys_values_plus_2 = num_keys_values + 2
    num_keys_values_plus_3 = num_keys_values + 3
    num_keys_values_plus_4 = num_keys_values + 4
    num_keys_values_plus_5 = num_keys_values + 5

    init_ids_len_1 = ort.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, device_id)
    init_history_len = ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, device_id)
    init_concat_text_len = ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, device_id)

    init_audio_start_ids = ort.OrtValue.ortvalue_from_numpy(np.array([[101]], dtype=np.int32), device_type, device_id)

    init_attention_mask_0 = ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, device_id)
    init_attention_mask_1 = ort.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, device_id)

    shape_keys = (ort_session_F._inputs_meta[0].shape[0], 1, ort_session_F._inputs_meta[0].shape[2], 0)
    shape_vals = (ort_session_F._inputs_meta[num_layers].shape[0], 1, 0, ort_session_F._inputs_meta[num_layers].shape[3])
    shape_embed = (1, 0, ort_session_F._inputs_meta[num_keys_values_plus_1].shape[2])
    shape_latent = (ort_session_H._inputs_meta[0].shape[0], 0, ort_session_H._inputs_meta[0].shape[2])

    init_past_keys_F = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape_keys, dtype=model_dtype_F), device_type, device_id)
    init_past_values_F = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape_vals, dtype=model_dtype_F), device_type, device_id)
    init_feat_embed = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape_embed, dtype=model_dtype_F), device_type, device_id)
    init_latent_pred = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape_latent, dtype=model_dtype_H), device_type, device_id)

    cfg_value_tensor = ort.OrtValue.ortvalue_from_numpy(np.array([cfg_value], dtype=model_dtype_G), device_type, device_id)
    cfg_value_minus = ort.OrtValue.ortvalue_from_numpy(np.array([1.0 - cfg_value], dtype=model_dtype_G), device_type, device_id)

    timesteps = fixed_timesteps - 1
    init_cfm_steps = ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int32), device_type, device_id)

    blank_segment = np.zeros((1, 1, int(out_sample_rate * blank_duration)), dtype=np.int16)

    input_feed_A = {}
    input_feed_B = {}
    input_feed_C = {}
    input_feed_D = {}
    input_feed_E = {}
    input_feed_F = {}
    input_feed_G = {}
    input_feed_H = {}

    input_feed_A[in_name_A] = init_audio_start_ids
    audio_start_embed = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]

    input_feed_D[in_name_D] = ort.OrtValue.ortvalue_from_numpy(
        np.zeros([1, ort_session_D._inputs_meta[0].shape[1], ort_session_D._inputs_meta[0].shape[2]], dtype=model_dtype_D),
        device_type,
        device_id,
    )
    init_feat_cond_0 = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)[0]

    input_feed_G[in_name_G[4]] = cfg_value_tensor
    input_feed_G[in_name_G[5]] = cfg_value_minus

    tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(args.voxcpm_dir))

    if args.prompt_audio:
        audio = read_audio_mono_int16(args.prompt_audio, in_sample_rate, max_prompt_audio_len)
        if use_audio_normalizer:
            audio = audio_normalizer(audio)
        audio = ort.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), device_type, device_id)
        use_prompt_audio = True
    else:
        use_prompt_audio = False
        prompt_text = None

    count_time = time.time()
    if use_prompt_audio:
        input_feed_B[in_name_B] = audio
        audio_feat = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)[0]

        input_feed_D[in_name_D] = audio_feat
        init_feat_cond = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)[0]

        if use_text_normalizer and prompt_text:
            prompt_text = text_normalizer.normalize(prompt_text)

        prompt_ids = np.array([tokenizer(prompt_text)], dtype=np.int32)
        prompt_text_len = int(prompt_ids.shape[-1])
        input_feed_A[in_name_A] = ort.OrtValue.ortvalue_from_numpy(prompt_ids, device_type, device_id)
        prompt_embed = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
    else:
        init_feat_cond = init_feat_cond_0
        prompt_text_len = 0
        prompt_embed = None
        audio_feat = None

    save_audio_out = []

    for sentence in texts:
        print(f"Convert to Speech: {sentence}")
        if use_text_normalizer:
            sentence = text_normalizer.normalize(sentence)

        target_ids = np.array([tokenizer(sentence)], dtype=np.int32)
        input_feed_A[in_name_A] = ort.OrtValue.ortvalue_from_numpy(target_ids, device_type, device_id)
        target_embed = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]

        if use_prompt_audio:
            input_feed_E[in_name_E[0]] = prompt_embed
            input_feed_E[in_name_E[1]] = target_embed
            target_embed, _ = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)

        input_feed_E[in_name_E[0]] = target_embed
        input_feed_E[in_name_E[1]] = audio_start_embed
        concat_embed, concat_text_len = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)

        if use_prompt_audio:
            input_feed_C[in_name_C] = audio_feat
            feat_embed = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)[0]

            input_feed_E[in_name_E[0]] = concat_embed
            input_feed_E[in_name_E[1]] = feat_embed
            concat_embed, ids_len = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
        else:
            feat_embed = init_feat_embed
            ids_len = concat_text_len

        concat_text_len_val = int(concat_text_len.numpy().item())
        ids_len_val = int(ids_len.numpy().item())
        max_len = min((concat_text_len_val - prompt_text_len) * decode_limit_factor + 10, generate_limit - ids_len_val)
        if max_len <= 0:
            print("Warning: max_len <= 0, skipping sentence.")
            continue

        input_feed_F[in_name_F[num_keys_values]] = init_history_len
        input_feed_F[in_name_F[num_keys_values_plus_1]] = feat_embed
        input_feed_F[in_name_F[num_keys_values_plus_2]] = concat_text_len
        input_feed_F[in_name_F[num_keys_values_plus_3]] = concat_embed
        input_feed_F[in_name_F[num_keys_values_plus_4]] = ids_len
        input_feed_F[in_name_F[num_keys_values_plus_5]] = init_attention_mask_1

        for i in range(num_layers):
            input_feed_F[in_name_F[i]] = init_past_keys_F
        for i in range(num_layers, num_keys_values):
            input_feed_F[in_name_F[i]] = init_past_values_F

        feat_cond = init_feat_cond

        if not args.streaming:
            save_latent = init_latent_pred if dynamic_shape_vae_decode else []

        num_decode = 0
        start_decode = time.time()

        while num_decode < max_len:
            all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)

            input_feed_G[in_name_G[0]] = init_cfm_steps
            input_feed_G[in_name_G[1]] = all_outputs_F[num_keys_values_plus_1]
            input_feed_G[in_name_G[2]] = all_outputs_F[num_keys_values_plus_2]
            input_feed_G[in_name_G[3]] = feat_cond

            for _ in range(timesteps):
                all_outputs_G = ort_session_G.run_with_ort_values(out_name_G, input_feed_G)
                input_feed_G[in_name_G[0]] = all_outputs_G[0]
                input_feed_G[in_name_G[1]] = all_outputs_G[1]

            latent_pred = all_outputs_G[1]

            if args.streaming:
                if num_decode < 1:
                    pre_latent_pred = latent_pred
                else:
                    input_feed_E[in_name_E[0]] = pre_latent_pred
                    input_feed_E[in_name_E[1]] = latent_pred
                    save_latent, _ = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                    input_feed_H[in_name_H] = save_latent
                    audio_out, _ = ort_session_H.run_with_ort_values(out_name_H, input_feed_H)
                    pre_latent_pred = latent_pred
                    audio_out = audio_out.numpy()
                    if num_decode > 1:
                        audio_out = audio_out[..., half_decode_len:]
                    save_audio_out.append(audio_out)
            else:
                if dynamic_shape_vae_decode:
                    input_feed_E[in_name_E[0]] = save_latent
                    input_feed_E[in_name_E[1]] = latent_pred
                    save_latent, _ = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                else:
                    save_latent.append(latent_pred)

            if num_decode >= min_seq_len:
                stop_id = int(all_outputs_F[num_keys_values_plus_3].numpy().item())
                if stop_id in STOP_TOKEN:
                    break

            input_feed_C[in_name_C] = latent_pred
            feat_embed = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)[0]

            input_feed_D[in_name_D] = latent_pred
            feat_cond = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)[0]

            input_feed_F.update(zip(in_name_F[:num_keys_values_plus_1], all_outputs_F))
            input_feed_F[in_name_F[num_keys_values_plus_1]] = feat_embed
            input_feed_F[in_name_F[num_keys_values_plus_3]] = feat_embed

            if num_decode < 1:
                input_feed_F[in_name_F[num_keys_values_plus_2]] = init_concat_text_len
                input_feed_F[in_name_F[num_keys_values_plus_4]] = init_ids_len_1
                input_feed_F[in_name_F[num_keys_values_plus_5]] = init_attention_mask_0

            num_decode += 1
            print(f"    Decode: {num_decode}")

        print(f"Decode Speed: {((num_decode + 1) / (time.time() - start_decode)):.3f} token/s")

        if not args.streaming:
            if dynamic_shape_vae_decode:
                input_feed_H[in_name_H] = save_latent
                audio_out, _ = ort_session_H.run_with_ort_values(out_name_H, input_feed_H)
                save_audio_out.append(audio_out.numpy())
            else:
                input_feed_E[in_name_E[0]] = save_latent[0]
                input_feed_E[in_name_E[1]] = save_latent[1]
                concat_latent, _ = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                input_feed_H[in_name_H] = concat_latent
                audio_out, _ = ort_session_H.run_with_ort_values(out_name_H, input_feed_H)
                save_audio_out.append(audio_out.numpy())
                for i in range(2, len(save_latent)):
                    input_feed_E[in_name_E[0]] = save_latent[i - 1]
                    input_feed_E[in_name_E[1]] = save_latent[i]
                    concat_latent, _ = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                    input_feed_H[in_name_H] = concat_latent
                    audio_out, _ = ort_session_H.run_with_ort_values(out_name_H, input_feed_H)
                    audio_out = audio_out.numpy()[..., half_decode_len:]
                    save_audio_out.append(audio_out)

        save_audio_out.append(blank_segment)

    if not save_audio_out:
        raise RuntimeError("No audio generated.")

    cost_time = time.time() - count_time
    audio_out = np.concatenate(save_audio_out, axis=-1).reshape(-1)
    if use_audio_normalizer:
        audio_out = audio_normalizer(audio_out)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sf.write(args.output, audio_out, out_sample_rate, format="WAVEX")

    total_audio_samples = audio_out.shape[-1] - blank_segment.shape[-1] * len(texts)
    total_audio_samples = max(total_audio_samples, 1)
    total_audio_duration = total_audio_samples / out_sample_rate
    rtf = cost_time / total_audio_duration if total_audio_duration > 0 else 0.0

    print("Generate Complete.")
    print(f"Saving to: {args.output}")
    print(f"Time Cost: {cost_time:.3f} Seconds")
    print(f"RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()
