"""VoxCPM 0.5B ONNX Engine.

This engine uses the 4-file ONNX architecture for VoxCPM 0.5B model.
Based on bluryar/VoxCPM-ONNX implementation.

Key differences from 1.5B:
- 4 ONNX files instead of 8
- 16kHz sample rate instead of 44.1kHz
- Different prefill/decode architecture
"""

from __future__ import annotations

import json
import os
from typing import Generator, List

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import LlamaTokenizerFast

from .base import BaseEngine

# Import utilities from infer.py
import infer as voxcpm_infer


# 0.5B model constants
SAMPLE_RATE_05B = 16000
CHUNK_SIZE = 640  # 40ms at 16kHz
AUDIO_START_TOKEN = 101


class VoxCPM05BEngine(BaseEngine):
    """VoxCPM 0.5B ONNX inference engine.
    
    Uses 4 ONNX files:
    - audio_vae_encoder.onnx: Encodes audio to latent (64-dim patches)
    - audio_vae_decoder.onnx: Decodes latent patches to audio
    - voxcpm_prefill.onnx: Initial prefill step
    - voxcpm_decode_step.onnx: Autoregressive decode step
    """

    REQUIRED_ONNX = [
        "audio_vae_encoder.onnx",
        "audio_vae_decoder.onnx",
        "voxcpm_prefill.onnx",
        "voxcpm_decode_step.onnx",
    ]

    def __init__(
        self,
        models_dir: str,
        voxcpm_dir: str,
        voices_file: str,
        onnx_config: str | None = None,
        max_threads: int = 0,
        text_normalizer: bool = True,
        audio_normalizer: bool = False,
    ) -> None:
        self.models_dir = models_dir
        self.voxcpm_dir = voxcpm_dir
        self.voices_file = voices_file
        self.onnx_config_path = onnx_config
        self.default_text_normalizer = text_normalizer
        self.default_audio_normalizer = audio_normalizer

        self.config = self._load_config()
        
        self.in_sample_rate = SAMPLE_RATE_05B
        self.out_sample_rate = SAMPLE_RATE_05B
        self.max_seq_len = int(self.config.get("max_seq_len", 512))
        self.min_seq_len = int(self.config.get("min_seq_len", 2))
        self.cfg_value = float(self.config.get("cfg_value", 2.5))
        self.random_seed = int(self.config.get("random_seed", 1))
        self.max_prompt_seconds = int(self.config.get("max_prompt_audio_seconds", 20))
        self.max_prompt_audio_len = self.max_prompt_seconds * self.in_sample_rate
        self.blank_duration = float(self.config.get("blank_duration", 0.1))

        self._text_normalizer = None
        if self.default_text_normalizer:
            if voxcpm_infer.TextNormalizer is None:
                raise RuntimeError(
                    f"TextNormalizer unavailable: {voxcpm_infer.TEXT_NORMALIZER_IMPORT_ERROR}"
                )
            self._text_normalizer = voxcpm_infer.TextNormalizer()

        self.device_type = "cpu"
        self.device_id = 0

        self._validate_models()
        self._init_sessions(max_threads)
        self.tokenizer = voxcpm_infer.mask_multichar_chinese_tokens(
            LlamaTokenizerFast.from_pretrained(self.voxcpm_dir)
        )

    def _load_config(self) -> dict:
        config = {
            "max_seq_len": 512,
            "min_seq_len": 2,
            "cfg_value": 2.5,
            "random_seed": 1,
            "max_prompt_audio_seconds": 20,
            "blank_duration": 0.1,
        }
        
        config_path = self.onnx_config_path
        if config_path is None:
            config_path = os.path.join(self.models_dir, "voxcpm_onnx_config.json")
        
        if config_path and os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                config.update(loaded)
        
        return config

    def _validate_models(self) -> None:
        missing = [
            name for name in self.REQUIRED_ONNX
            if not os.path.isfile(os.path.join(self.models_dir, name))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing 0.5B ONNX files in {self.models_dir}: {', '.join(missing)}"
            )

    @property
    def sample_rate(self) -> int:
        return self.out_sample_rate

    @property
    def bit_depth(self) -> int:
        return 16

    @property
    def channels(self) -> int:
        return 1

    def _init_sessions(self, max_threads: int) -> None:
        session_opts = ort.SessionOptions()
        session_opts.log_severity_level = 4
        session_opts.inter_op_num_threads = max_threads
        session_opts.intra_op_num_threads = max_threads
        session_opts.enable_cpu_mem_arena = True
        session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]

        def model_path(name: str) -> str:
            return os.path.join(self.models_dir, name)

        # Load all sessions
        self.vae_encoder = ort.InferenceSession(
            model_path("audio_vae_encoder.onnx"), sess_options=session_opts, providers=providers
        )
        self.vae_decoder = ort.InferenceSession(
            model_path("audio_vae_decoder.onnx"), sess_options=session_opts, providers=providers
        )
        self.prefill = ort.InferenceSession(
            model_path("voxcpm_prefill.onnx"), sess_options=session_opts, providers=providers
        )
        self.decode_step = ort.InferenceSession(
            model_path("voxcpm_decode_step.onnx"), sess_options=session_opts, providers=providers
        )

    def _encode_audio(self, audio: np.ndarray) -> np.ndarray:
        """Encode audio to latent patches using VAE encoder.
        
        Input: audio [samples] int16
        Output: latent [batch, 64, num_patches] float32
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, 1, -1)
        elif audio.ndim == 2:
            audio = audio.reshape(1, audio.shape[0], audio.shape[1])
        
        audio_float = audio.astype(np.float32) / 32768.0
        outputs = self.vae_encoder.run(None, {"audio_data": audio_float})
        return outputs[0]  # z: [batch, 64, num_patches]

    def _decode_latents(self, latents: np.ndarray) -> np.ndarray:
        """Decode latent patches to audio using VAE decoder.
        
        Input: latents [batch, 64, num_patches] float32
        Output: audio [samples] int16
        """
        outputs = self.vae_decoder.run(None, {"z": latents})
        audio = outputs[0]  # [batch, 1, samples]
        audio = np.clip(audio * 32768.0, -32768.0, 32767.0).astype(np.int16)
        return audio.reshape(-1)

    def _run_prefill(
        self,
        text_tokens: np.ndarray,
        audio_feat: np.ndarray | None = None,
    ) -> dict:
        """Run prefill step.
        
        Inputs:
            text: [batch, seq_len] int64
            text_mask: [batch, seq_len] int32
            feat: [batch, seq_len, 2, 64] float32 (audio features, zero-padded)
            feat_mask: [batch, seq_len] int32
        
        Outputs:
            dit_hidden, base_next_keys/values, residual_next_keys/values, prefix_feat_cond
        """
        batch_size = text_tokens.shape[0]
        seq_len = text_tokens.shape[1]
        
        # Text and mask
        text = text_tokens.astype(np.int64)
        text_mask = np.ones((batch_size, seq_len), dtype=np.int32)
        
        # Audio features - reshape from [batch, 64, patches] to [batch, patches, 2, 64]
        if audio_feat is not None and audio_feat.size > 0:
            # audio_feat is [batch, 64, num_patches]
            num_patches = audio_feat.shape[2]
            # Reshape to [batch, num_patches, 2, 64] - split 64 into 2x64? No, it's already 64
            # Actually we need to pad to seq_len and reshape
            feat = np.zeros((batch_size, seq_len, 2, 64), dtype=np.float32)
            feat_mask = np.zeros((batch_size, seq_len), dtype=np.int32)
            
            # Transpose [batch, 64, patches] -> [batch, patches, 64]
            audio_feat_t = audio_feat.transpose(0, 2, 1)  # [batch, patches, 64]
            
            # Now we need to expand 64 to 2x64 = 128? Or is it 2 separate patches per position?
            # Looking at the model input: [batch, seq_len, 2, 64]
            # This suggests 2 "channels" of 64-dim features per sequence position
            # For each audio patch, we'll duplicate it into both channels
            for i in range(min(num_patches, seq_len)):
                feat[:, i, 0, :] = audio_feat_t[:, i, :]
                feat[:, i, 1, :] = audio_feat_t[:, i, :]
                feat_mask[:, i] = 1
        else:
            feat = np.zeros((batch_size, seq_len, 2, 64), dtype=np.float32)
            feat_mask = np.zeros((batch_size, seq_len), dtype=np.int32)
        
        outputs = self.prefill.run(
            None,
            {
                "text": text,
                "text_mask": text_mask,
                "feat": feat,
                "feat_mask": feat_mask,
            }
        )
        
        return {
            "dit_hidden": outputs[0],
            "base_next_keys": outputs[1],
            "base_next_values": outputs[2],
            "residual_next_keys": outputs[3],
            "residual_next_values": outputs[4],
            "prefix_feat_cond": outputs[5],
        }

    def _run_decode_step(self, state: dict, cfg_value: float) -> tuple[np.ndarray, bool, dict]:
        """Run one decode step.
        
        Inputs:
            dit_hidden: [batch, 1024]
            base_next_keys: [batch, 24, 2, past_seq, 64]
            base_next_values: [batch, 24, 2, past_seq, 64]
            residual_next_keys: [batch, 6, 2, past_seq, 64]
            residual_next_values: [batch, 6, 2, past_seq, 64]
            prefix_feat_cond: [batch, 2, 64]
            noise: [batch, 2, 64]
            cfg_value: scalar float
        
        Returns:
            (pred_feat, stop_flag, new_state)
        """
        batch_size = state["dit_hidden"].shape[0]
        
        # Generate random noise
        noise = np.random.randn(batch_size, 2, 64).astype(np.float32)
        
        outputs = self.decode_step.run(
            None,
            {
                "dit_hidden": state["dit_hidden"],
                "base_next_keys": state["base_next_keys"],
                "base_next_values": state["base_next_values"],
                "residual_next_keys": state["residual_next_keys"],
                "residual_next_values": state["residual_next_values"],
                "prefix_feat_cond": state["prefix_feat_cond"],
                "noise": noise,
                "cfg_value": np.array(cfg_value, dtype=np.float32),
            }
        )
        
        pred_feat = outputs[0]  # [batch, 2, 64]
        stop_flag = outputs[6]  # [batch]
        
        new_state = {
            "dit_hidden": outputs[1],
            "base_next_keys": outputs[2],
            "base_next_values": outputs[3],
            "residual_next_keys": outputs[4],
            "residual_next_values": outputs[5],
            "prefix_feat_cond": state["prefix_feat_cond"],  # Keep original
        }
        
        return pred_feat, bool(stop_flag[0]), new_state

    def synthesize(
        self,
        texts: List[str],
        voice: str | None = None,
        prompt_audio: str | None = None,
        prompt_text: str | None = None,
        voices_file: str | None = None,
        cfg_value: float | None = None,
        fixed_timesteps: int | None = None,
        seed: int | None = None,
        streaming: bool = False,
        use_text_normalizer: bool | None = None,
        use_audio_normalizer: bool | None = None,
    ) -> tuple[np.ndarray, int]:
        if not texts:
            raise ValueError("No text provided")

        voices_file = voices_file or self.voices_file
        if voice:
            if prompt_audio or prompt_text:
                raise ValueError("Use either voice or prompt_audio/prompt_text")
            prompt_audio, prompt_text = voxcpm_infer.resolve_voice_prompt(voices_file, voice)
        
        if (prompt_audio is None) != (prompt_text is None):
            raise ValueError("prompt_audio and prompt_text must both be provided or both be omitted")

        cfg_value = cfg_value if cfg_value is not None else self.cfg_value
        seed = int(seed if seed is not None else self.random_seed)
        np.random.seed(seed)

        use_text_normalizer = self.default_text_normalizer if use_text_normalizer is None else use_text_normalizer
        if use_text_normalizer and self._text_normalizer is None:
            raise RuntimeError("TextNormalizer unavailable")
        use_audio_normalizer = self.default_audio_normalizer if use_audio_normalizer is None else use_audio_normalizer

        # Prepare prompt audio features
        audio_feat = None
        if prompt_audio:
            audio = voxcpm_infer.read_audio_mono_int16(prompt_audio, self.in_sample_rate, self.max_prompt_audio_len)
            if use_audio_normalizer:
                audio = voxcpm_infer.audio_normalizer(audio)
            audio_feat = self._encode_audio(audio)  # [1, 64, patches]
            
            if use_text_normalizer and prompt_text:
                prompt_text = self._text_normalizer.normalize(prompt_text)

        all_audio = []
        blank_segment = np.zeros(int(self.out_sample_rate * self.blank_duration), dtype=np.int16)

        for sentence in texts:
            print(f"Convert to Speech: {sentence}")
            
            if use_text_normalizer:
                sentence = self._text_normalizer.normalize(sentence)

            # Tokenize - combine prompt text and target text
            if prompt_text:
                full_text = f"{prompt_text} {sentence}"
            else:
                full_text = sentence
            
            tokens = self.tokenizer(full_text)
            # CRITICAL: Append AUDIO_START_TOKEN for model to know when audio generation should start/stop
            tokens = tokens + [AUDIO_START_TOKEN]
            text_tokens = np.array([tokens], dtype=np.int64)

            # Run prefill
            state = self._run_prefill(text_tokens, audio_feat)

            # Autoregressive decode loop
            latent_list = []
            
            for step in range(self.max_seq_len):
                pred_feat, stop_flag, state = self._run_decode_step(state, cfg_value)
                latent_list.append(pred_feat)
                print(f"    Decode: {step + 1}")
                
                if stop_flag and step >= self.min_seq_len:
                    break

            # Stack latents: list of [batch, 2, 64] -> [batch, 64, num_steps*2]
            if latent_list:
                # Each pred_feat is [1, 2, 64], we need [1, 64, num_patches]
                # Reshape: stack along axis 1, then transpose
                stacked = np.concatenate(latent_list, axis=1)  # [1, num_steps*2, 64]
                latents = stacked.transpose(0, 2, 1)  # [1, 64, num_steps*2]
                
                audio_out = self._decode_latents(latents)
                all_audio.append(audio_out)
                print(f"Decode Speed: {len(latent_list) / 1.0:.3f} token/s")  # Placeholder
            
            all_audio.append(blank_segment)

        if not all_audio:
            raise RuntimeError("No audio generated")

        audio_out = np.concatenate(all_audio)
        if use_audio_normalizer:
            audio_out = voxcpm_infer.audio_normalizer(audio_out)

        return audio_out, self.out_sample_rate

    def synthesize_stream(
        self,
        texts: List[str],
        voice: str | None = None,
        prompt_audio: str | None = None,
        prompt_text: str | None = None,
        voices_file: str | None = None,
        cfg_value: float | None = None,
        fixed_timesteps: int | None = None,
        seed: int | None = None,
        use_text_normalizer: bool | None = None,
        use_audio_normalizer: bool | None = None,
        chunk_tokens: int = 4,
    ) -> Generator[bytes, None, None]:
        """Streaming synthesis - yields PCM chunks."""
        if not texts:
            raise ValueError("No text provided")

        voices_file = voices_file or self.voices_file
        if voice:
            if prompt_audio or prompt_text:
                raise ValueError("Use either voice or prompt_audio/prompt_text")
            prompt_audio, prompt_text = voxcpm_infer.resolve_voice_prompt(voices_file, voice)
        
        if (prompt_audio is None) != (prompt_text is None):
            raise ValueError("prompt_audio and prompt_text must both be provided or both be omitted")

        cfg_value = cfg_value if cfg_value is not None else self.cfg_value
        seed = int(seed if seed is not None else self.random_seed)
        np.random.seed(seed)

        use_text_normalizer = self.default_text_normalizer if use_text_normalizer is None else use_text_normalizer

        audio_feat = None
        if prompt_audio:
            audio = voxcpm_infer.read_audio_mono_int16(prompt_audio, self.in_sample_rate, self.max_prompt_audio_len)
            audio_feat = self._encode_audio(audio)
            
            if use_text_normalizer and prompt_text:
                prompt_text = self._text_normalizer.normalize(prompt_text)

        blank_segment = np.zeros(int(self.out_sample_rate * self.blank_duration), dtype=np.int16)

        for sentence in texts:
            if use_text_normalizer:
                sentence = self._text_normalizer.normalize(sentence)

            if prompt_text:
                full_text = f"{prompt_text} {sentence}"
            else:
                full_text = sentence
            
            tokens = self.tokenizer(full_text)
            # CRITICAL: Append AUDIO_START_TOKEN for model to know when audio generation should start/stop
            tokens = tokens + [AUDIO_START_TOKEN]
            text_tokens = np.array([tokens], dtype=np.int64)

            state = self._run_prefill(text_tokens, audio_feat)

            latent_buffer = []
            
            for step in range(self.max_seq_len):
                pred_feat, stop_flag, state = self._run_decode_step(state, cfg_value)
                latent_buffer.append(pred_feat)
                
                if len(latent_buffer) >= chunk_tokens:
                    stacked = np.concatenate(latent_buffer, axis=1)
                    latents = stacked.transpose(0, 2, 1)
                    audio_chunk = self._decode_latents(latents)
                    yield audio_chunk.tobytes()
                    latent_buffer = []
                
                if stop_flag and step >= self.min_seq_len:
                    break

            if latent_buffer:
                stacked = np.concatenate(latent_buffer, axis=1)
                latents = stacked.transpose(0, 2, 1)
                audio_chunk = self._decode_latents(latents)
                yield audio_chunk.tobytes()

            yield blank_segment.tobytes()
