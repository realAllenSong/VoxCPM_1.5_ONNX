"""VoxCPM 1.5B ONNX Engine.

This engine uses the 8-file ONNX architecture for VoxCPM 1.5B model.
Refactored from the original api_server.py VoxCPMEngine class.

Advanced Features:
- True streaming output (low TTFB)
- Prompt Cache for multi-sentence consistency
- Retry mechanism for badcase handling
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple, Any

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import LlamaTokenizerFast

from .base import BaseEngine

# Import utilities from infer.py
import infer as voxcpm_infer


@dataclass
class PromptCache:
    """Cache for prompt audio features and text, enabling multi-sentence generation
    with consistent voice across multiple calls.
    
    Usage:
        cache = engine.build_prompt_cache(prompt_audio, prompt_text)
        for sentence in sentences:
            for chunk in engine.synthesize_with_cache_stream(sentence, cache):
                yield chunk
            cache = engine.merge_prompt_cache(cache, generated_audio_feat)
    """
    audio_feat: Any = None  # OrtValue or None
    feat_cond: Any = None   # OrtValue
    prompt_embed: Any = None  # OrtValue or None
    prompt_text_len: int = 0
    generated_audio_feats: list = field(default_factory=list)  # List of generated latents
    use_prompt_audio: bool = False


class VoxCPM15BEngine(BaseEngine):
    """VoxCPM 1.5B ONNX inference engine.
    
    Uses 8 ONNX files:
    - VoxCPM_Text_Embed.onnx
    - VoxCPM_VAE_Encoder.onnx
    - VoxCPM_Feat_Encoder.onnx
    - VoxCPM_Feat_Cond.onnx
    - VoxCPM_Concat.onnx
    - VoxCPM_Main.onnx
    - VoxCPM_Feat_Decoder.onnx
    - VoxCPM_VAE_Decoder.onnx
    """

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

        config, _ = voxcpm_infer.load_onnx_config(self.models_dir, self.onnx_config_path)
        self.config = config

        self.max_seq_len = int(config["max_seq_len"])
        self.min_seq_len = int(config["min_seq_len"])
        self.decode_limit_factor = int(config["decode_limit_factor"])
        self.in_sample_rate = int(config["in_sample_rate"])
        self.out_sample_rate = int(config["out_sample_rate"])
        self.fixed_timesteps = int(config["fixed_timesteps"])
        self.cfg_value = float(config["cfg_value"])
        self.random_seed = int(config["random_seed"])
        self.max_prompt_seconds = int(config.get("max_prompt_audio_seconds", 20))
        self.max_prompt_audio_len = self.max_prompt_seconds * self.in_sample_rate
        self.half_decode_len = int(config.get("half_decode_len", 7056))
        self.blank_duration = float(config.get("blank_duration", 0.1))

        self._text_normalizer = None
        if self.default_text_normalizer:
            if voxcpm_infer.TextNormalizer is None:
                raise RuntimeError(
                    "TextNormalizer unavailable. Install dependencies: pip install wetext regex inflect "
                    f"(error: {voxcpm_infer.TEXT_NORMALIZER_IMPORT_ERROR})"
                )
            self._text_normalizer = voxcpm_infer.TextNormalizer()

        self.device_type = "cpu"
        self.device_id = 0

        self._init_sessions(max_threads)
        self.tokenizer = voxcpm_infer.mask_multichar_chinese_tokens(
            LlamaTokenizerFast.from_pretrained(self.voxcpm_dir)
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
        run_options = ort.RunOptions()

        session_opts.log_severity_level = 4
        session_opts.log_verbosity_level = 4
        run_options.log_severity_level = 4
        run_options.log_verbosity_level = 4

        session_opts.inter_op_num_threads = max_threads
        session_opts.intra_op_num_threads = max_threads
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

        def model_path(name: str) -> str:
            return os.path.join(self.models_dir, name)

        self.ort_session_A = ort.InferenceSession(model_path("VoxCPM_Text_Embed.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
        self.in_name_A = self.ort_session_A.get_inputs()[0].name
        self.out_name_A = [self.ort_session_A.get_outputs()[0].name]

        self.ort_session_B = ort.InferenceSession(model_path("VoxCPM_VAE_Encoder.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
        self.in_name_B = self.ort_session_B.get_inputs()[0].name
        self.out_name_B = [self.ort_session_B.get_outputs()[0].name]

        self.ort_session_C = ort.InferenceSession(model_path("VoxCPM_Feat_Encoder.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
        self.in_name_C = self.ort_session_C.get_inputs()[0].name
        self.out_name_C = [self.ort_session_C.get_outputs()[0].name]

        self.ort_session_D = ort.InferenceSession(model_path("VoxCPM_Feat_Cond.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
        model_dtype_D = self.ort_session_D._inputs_meta[0].type
        self.model_dtype_D = np.float16 if "float16" in model_dtype_D else np.float32
        self.in_name_D = self.ort_session_D.get_inputs()[0].name
        self.out_name_D = [self.ort_session_D.get_outputs()[0].name]

        self.ort_session_E = ort.InferenceSession(model_path("VoxCPM_Concat.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
        self.in_name_E = [item.name for item in self.ort_session_E.get_inputs()]
        self.out_name_E = [item.name for item in self.ort_session_E.get_outputs()]

        self.ort_session_F = ort.InferenceSession(model_path("VoxCPM_Main.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
        self.in_name_F = [item.name for item in self.ort_session_F.get_inputs()]
        self.out_name_F = [item.name for item in self.ort_session_F.get_outputs()]
        amount_of_outputs_F = len(self.out_name_F)

        model_dtype_F = self.ort_session_F._inputs_meta[0].type
        self.model_dtype_F = np.float16 if "float16" in model_dtype_F else np.float32

        self.ort_session_G = ort.InferenceSession(model_path("VoxCPM_Feat_Decoder.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
        model_dtype_G = self.ort_session_G._inputs_meta[2].type
        self.model_dtype_G = np.float16 if "float16" in model_dtype_G else np.float32
        self.in_name_G = [item.name for item in self.ort_session_G.get_inputs()]
        self.out_name_G = [item.name for item in self.ort_session_G.get_outputs()]

        self.ort_session_H = ort.InferenceSession(model_path("VoxCPM_VAE_Decoder.onnx"), sess_options=session_opts, providers=providers, provider_options=provider_options)
        model_dtype_H = self.ort_session_H._inputs_meta[0].type
        self.model_dtype_H = np.float16 if "float16" in model_dtype_H else np.float32

        shape_value_in_H = self.ort_session_H._inputs_meta[0].shape[1]
        self.dynamic_shape_vae_decode = isinstance(shape_value_in_H, str)

        self.in_name_H = self.ort_session_H.get_inputs()[0].name
        self.out_name_H = [item.name for item in self.ort_session_H.get_outputs()]

        self.generate_limit = self.max_seq_len - 1
        self.num_keys_values = amount_of_outputs_F - 4
        self.num_layers = self.num_keys_values // 2
        self.num_keys_values_plus_1 = self.num_keys_values + 1
        self.num_keys_values_plus_2 = self.num_keys_values + 2
        self.num_keys_values_plus_3 = self.num_keys_values + 3
        self.num_keys_values_plus_4 = self.num_keys_values + 4
        self.num_keys_values_plus_5 = self.num_keys_values + 5

        self.init_ids_len_1 = ort.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), self.device_type, self.device_id)
        self.init_history_len = ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), self.device_type, self.device_id)
        self.init_concat_text_len = ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), self.device_type, self.device_id)
        self.init_audio_start_ids = ort.OrtValue.ortvalue_from_numpy(np.array([[101]], dtype=np.int32), self.device_type, self.device_id)
        self.init_attention_mask_0 = ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), self.device_type, self.device_id)
        self.init_attention_mask_1 = ort.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), self.device_type, self.device_id)

        shape_keys = (self.ort_session_F._inputs_meta[0].shape[0], 1, self.ort_session_F._inputs_meta[0].shape[2], 0)
        shape_vals = (self.ort_session_F._inputs_meta[self.num_layers].shape[0], 1, 0, self.ort_session_F._inputs_meta[self.num_layers].shape[3])
        shape_embed = (1, 0, self.ort_session_F._inputs_meta[self.num_keys_values_plus_1].shape[2])
        shape_latent = (self.ort_session_H._inputs_meta[0].shape[0], 0, self.ort_session_H._inputs_meta[0].shape[2])

        self.init_past_keys_F = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape_keys, dtype=self.model_dtype_F), self.device_type, self.device_id)
        self.init_past_values_F = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape_vals, dtype=self.model_dtype_F), self.device_type, self.device_id)
        self.init_feat_embed = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape_embed, dtype=self.model_dtype_F), self.device_type, self.device_id)
        self.init_latent_pred = ort.OrtValue.ortvalue_from_numpy(np.zeros(shape_latent, dtype=self.model_dtype_H), self.device_type, self.device_id)

        input_feed_A = {self.in_name_A: self.init_audio_start_ids}
        self.audio_start_embed = self.ort_session_A.run_with_ort_values(self.out_name_A, input_feed_A)[0]

        input_feed_D = {
            self.in_name_D: ort.OrtValue.ortvalue_from_numpy(
                np.zeros([1, self.ort_session_D._inputs_meta[0].shape[1], self.ort_session_D._inputs_meta[0].shape[2]], dtype=self.model_dtype_D),
                self.device_type,
                self.device_id,
            )
        }
        self.init_feat_cond_0 = self.ort_session_D.run_with_ort_values(self.out_name_D, input_feed_D)[0]

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

        voxcpm_infer.ensure_paths(self.models_dir, self.voxcpm_dir, prompt_audio)

        cfg_value = float(cfg_value if cfg_value is not None else self.cfg_value)
        fixed_timesteps = int(fixed_timesteps if fixed_timesteps is not None else self.fixed_timesteps)
        seed = int(seed if seed is not None else self.random_seed)
        ort.set_seed(seed)

        use_text_normalizer = self.default_text_normalizer if use_text_normalizer is None else use_text_normalizer
        if use_text_normalizer and self._text_normalizer is None:
            raise RuntimeError("TextNormalizer unavailable")
        use_audio_normalizer = self.default_audio_normalizer if use_audio_normalizer is None else use_audio_normalizer

        cfg_value_tensor = ort.OrtValue.ortvalue_from_numpy(np.array([cfg_value], dtype=self.model_dtype_G), self.device_type, self.device_id)
        cfg_value_minus = ort.OrtValue.ortvalue_from_numpy(np.array([1.0 - cfg_value], dtype=self.model_dtype_G), self.device_type, self.device_id)

        timesteps = fixed_timesteps - 1
        init_cfm_steps = ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int32), self.device_type, self.device_id)
        blank_segment = np.zeros((1, 1, int(self.out_sample_rate * self.blank_duration)), dtype=np.int16)

        input_feed_A = {}
        input_feed_B = {}
        input_feed_C = {}
        input_feed_D = {}
        input_feed_E = {}
        input_feed_F = {}
        input_feed_G = {}
        input_feed_H = {}

        input_feed_G[self.in_name_G[4]] = cfg_value_tensor
        input_feed_G[self.in_name_G[5]] = cfg_value_minus

        if prompt_audio:
            audio = voxcpm_infer.read_audio_mono_int16(prompt_audio, self.in_sample_rate, self.max_prompt_audio_len)
            if use_audio_normalizer:
                audio = voxcpm_infer.audio_normalizer(audio)
            audio = ort.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), self.device_type, self.device_id)
            use_prompt_audio = True
        else:
            use_prompt_audio = False
            prompt_text = None

        if use_prompt_audio:
            input_feed_B[self.in_name_B] = audio
            audio_feat = self.ort_session_B.run_with_ort_values(self.out_name_B, input_feed_B)[0]

            input_feed_D[self.in_name_D] = audio_feat
            init_feat_cond = self.ort_session_D.run_with_ort_values(self.out_name_D, input_feed_D)[0]

            if use_text_normalizer and prompt_text:
                prompt_text = self._text_normalizer.normalize(prompt_text)

            prompt_ids = np.array([self.tokenizer(prompt_text)], dtype=np.int32)
            prompt_text_len = int(prompt_ids.shape[-1])
            input_feed_A[self.in_name_A] = ort.OrtValue.ortvalue_from_numpy(prompt_ids, self.device_type, self.device_id)
            prompt_embed = self.ort_session_A.run_with_ort_values(self.out_name_A, input_feed_A)[0]
        else:
            init_feat_cond = self.init_feat_cond_0
            prompt_text_len = 0
            prompt_embed = None
            audio_feat = None

        save_audio_out = []

        for sentence in texts:
            if use_text_normalizer:
                sentence = self._text_normalizer.normalize(sentence)

            target_ids = np.array([self.tokenizer(sentence)], dtype=np.int32)
            input_feed_A[self.in_name_A] = ort.OrtValue.ortvalue_from_numpy(target_ids, self.device_type, self.device_id)
            target_embed = self.ort_session_A.run_with_ort_values(self.out_name_A, input_feed_A)[0]

            if use_prompt_audio:
                input_feed_E[self.in_name_E[0]] = prompt_embed
                input_feed_E[self.in_name_E[1]] = target_embed
                target_embed, _ = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)

            input_feed_E[self.in_name_E[0]] = target_embed
            input_feed_E[self.in_name_E[1]] = self.audio_start_embed
            concat_embed, concat_text_len = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)

            if use_prompt_audio:
                input_feed_C[self.in_name_C] = audio_feat
                feat_embed = self.ort_session_C.run_with_ort_values(self.out_name_C, input_feed_C)[0]

                input_feed_E[self.in_name_E[0]] = concat_embed
                input_feed_E[self.in_name_E[1]] = feat_embed
                concat_embed, ids_len = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)
            else:
                feat_embed = self.init_feat_embed
                ids_len = concat_text_len

            concat_text_len_val = int(concat_text_len.numpy().item())
            ids_len_val = int(ids_len.numpy().item())
            max_len = min((concat_text_len_val - prompt_text_len) * self.decode_limit_factor + 10, self.generate_limit - ids_len_val)
            if max_len <= 0:
                continue

            input_feed_F[self.in_name_F[self.num_keys_values]] = self.init_history_len
            input_feed_F[self.in_name_F[self.num_keys_values_plus_1]] = feat_embed
            input_feed_F[self.in_name_F[self.num_keys_values_plus_2]] = concat_text_len
            input_feed_F[self.in_name_F[self.num_keys_values_plus_3]] = concat_embed
            input_feed_F[self.in_name_F[self.num_keys_values_plus_4]] = ids_len
            input_feed_F[self.in_name_F[self.num_keys_values_plus_5]] = self.init_attention_mask_1

            for i in range(self.num_layers):
                input_feed_F[self.in_name_F[i]] = self.init_past_keys_F
            for i in range(self.num_layers, self.num_keys_values):
                input_feed_F[self.in_name_F[i]] = self.init_past_values_F

            feat_cond = init_feat_cond

            if not streaming:
                save_latent = self.init_latent_pred if self.dynamic_shape_vae_decode else []

            num_decode = 0
            while num_decode < max_len:
                all_outputs_F = self.ort_session_F.run_with_ort_values(self.out_name_F, input_feed_F)

                input_feed_G[self.in_name_G[0]] = init_cfm_steps
                input_feed_G[self.in_name_G[1]] = all_outputs_F[self.num_keys_values_plus_1]
                input_feed_G[self.in_name_G[2]] = all_outputs_F[self.num_keys_values_plus_2]
                input_feed_G[self.in_name_G[3]] = feat_cond

                for _ in range(timesteps):
                    all_outputs_G = self.ort_session_G.run_with_ort_values(self.out_name_G, input_feed_G)
                    input_feed_G[self.in_name_G[0]] = all_outputs_G[0]
                    input_feed_G[self.in_name_G[1]] = all_outputs_G[1]

                latent_pred = all_outputs_G[1]

                if streaming:
                    if num_decode < 1:
                        pre_latent_pred = latent_pred
                    else:
                        input_feed_E[self.in_name_E[0]] = pre_latent_pred
                        input_feed_E[self.in_name_E[1]] = latent_pred
                        save_latent, _ = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)
                        input_feed_H[self.in_name_H] = save_latent
                        audio_out, _ = self.ort_session_H.run_with_ort_values(self.out_name_H, input_feed_H)
                        pre_latent_pred = latent_pred
                        audio_out = audio_out.numpy()
                        if num_decode > 1:
                            audio_out = audio_out[..., self.half_decode_len:]
                        save_audio_out.append(audio_out)
                else:
                    if self.dynamic_shape_vae_decode:
                        input_feed_E[self.in_name_E[0]] = save_latent
                        input_feed_E[self.in_name_E[1]] = latent_pred
                        save_latent, _ = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)
                    else:
                        save_latent.append(latent_pred)

                if num_decode >= self.min_seq_len:
                    stop_id = int(all_outputs_F[self.num_keys_values_plus_3].numpy().item())
                    if stop_id in voxcpm_infer.STOP_TOKEN:
                        break

                input_feed_C[self.in_name_C] = latent_pred
                feat_embed = self.ort_session_C.run_with_ort_values(self.out_name_C, input_feed_C)[0]

                input_feed_D[self.in_name_D] = latent_pred
                feat_cond = self.ort_session_D.run_with_ort_values(self.out_name_D, input_feed_D)[0]

                input_feed_F.update(zip(self.in_name_F[:self.num_keys_values_plus_1], all_outputs_F))
                input_feed_F[self.in_name_F[self.num_keys_values_plus_1]] = feat_embed
                input_feed_F[self.in_name_F[self.num_keys_values_plus_3]] = feat_embed

                if num_decode < 1:
                    input_feed_F[self.in_name_F[self.num_keys_values_plus_2]] = self.init_concat_text_len
                    input_feed_F[self.in_name_F[self.num_keys_values_plus_4]] = self.init_ids_len_1
                    input_feed_F[self.in_name_F[self.num_keys_values_plus_5]] = self.init_attention_mask_0

                num_decode += 1

            if not streaming:
                if self.dynamic_shape_vae_decode:
                    input_feed_H[self.in_name_H] = save_latent
                    audio_out, _ = self.ort_session_H.run_with_ort_values(self.out_name_H, input_feed_H)
                    save_audio_out.append(audio_out.numpy())
                else:
                    input_feed_E[self.in_name_E[0]] = save_latent[0]
                    input_feed_E[self.in_name_E[1]] = save_latent[1]
                    concat_latent, _ = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)
                    input_feed_H[self.in_name_H] = concat_latent
                    audio_out, _ = self.ort_session_H.run_with_ort_values(self.out_name_H, input_feed_H)
                    save_audio_out.append(audio_out.numpy())
                    for i in range(2, len(save_latent)):
                        input_feed_E[self.in_name_E[0]] = save_latent[i - 1]
                        input_feed_E[self.in_name_E[1]] = save_latent[i]
                        concat_latent, _ = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)
                        input_feed_H[self.in_name_H] = concat_latent
                        audio_out, _ = self.ort_session_H.run_with_ort_values(self.out_name_H, input_feed_H)
                        audio_out = audio_out.numpy()[..., self.half_decode_len:]
                        save_audio_out.append(audio_out)

            save_audio_out.append(blank_segment)

        if not save_audio_out:
            raise RuntimeError("No audio generated")

        audio_out = np.concatenate(save_audio_out, axis=-1).reshape(-1)
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
        """Streaming synthesis - yields PCM chunks as they are generated."""
        if not texts:
            raise ValueError("No text provided")

        voices_file = voices_file or self.voices_file
        if voice:
            if prompt_audio or prompt_text:
                raise ValueError("Use either voice or prompt_audio/prompt_text")
            prompt_audio, prompt_text = voxcpm_infer.resolve_voice_prompt(voices_file, voice)
        if (prompt_audio is None) != (prompt_text is None):
            raise ValueError("prompt_audio and prompt_text must both be provided or both be omitted")

        voxcpm_infer.ensure_paths(self.models_dir, self.voxcpm_dir, prompt_audio)

        cfg_value = float(cfg_value if cfg_value is not None else self.cfg_value)
        fixed_timesteps = int(fixed_timesteps if fixed_timesteps is not None else self.fixed_timesteps)
        seed = int(seed if seed is not None else self.random_seed)
        ort.set_seed(seed)

        use_text_normalizer = self.default_text_normalizer if use_text_normalizer is None else use_text_normalizer
        if use_text_normalizer and self._text_normalizer is None:
            raise RuntimeError("TextNormalizer unavailable")
        use_audio_normalizer = self.default_audio_normalizer if use_audio_normalizer is None else use_audio_normalizer

        cfg_value_tensor = ort.OrtValue.ortvalue_from_numpy(np.array([cfg_value], dtype=self.model_dtype_G), self.device_type, self.device_id)
        cfg_value_minus = ort.OrtValue.ortvalue_from_numpy(np.array([1.0 - cfg_value], dtype=self.model_dtype_G), self.device_type, self.device_id)

        timesteps = fixed_timesteps - 1
        init_cfm_steps = ort.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int32), self.device_type, self.device_id)
        blank_segment = np.zeros((1, 1, int(self.out_sample_rate * self.blank_duration)), dtype=np.int16)

        input_feed_A = {}
        input_feed_B = {}
        input_feed_C = {}
        input_feed_D = {}
        input_feed_E = {}
        input_feed_F = {}
        input_feed_G = {}
        input_feed_H = {}

        input_feed_G[self.in_name_G[4]] = cfg_value_tensor
        input_feed_G[self.in_name_G[5]] = cfg_value_minus

        if prompt_audio:
            audio = voxcpm_infer.read_audio_mono_int16(prompt_audio, self.in_sample_rate, self.max_prompt_audio_len)
            if use_audio_normalizer:
                audio = voxcpm_infer.audio_normalizer(audio)
            audio = ort.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), self.device_type, self.device_id)
            use_prompt_audio = True
        else:
            use_prompt_audio = False
            prompt_text = None

        if use_prompt_audio:
            input_feed_B[self.in_name_B] = audio
            audio_feat = self.ort_session_B.run_with_ort_values(self.out_name_B, input_feed_B)[0]

            input_feed_D[self.in_name_D] = audio_feat
            init_feat_cond = self.ort_session_D.run_with_ort_values(self.out_name_D, input_feed_D)[0]

            if use_text_normalizer and prompt_text:
                prompt_text = self._text_normalizer.normalize(prompt_text)

            prompt_ids = np.array([self.tokenizer(prompt_text)], dtype=np.int32)
            prompt_text_len = int(prompt_ids.shape[-1])
            input_feed_A[self.in_name_A] = ort.OrtValue.ortvalue_from_numpy(prompt_ids, self.device_type, self.device_id)
            prompt_embed = self.ort_session_A.run_with_ort_values(self.out_name_A, input_feed_A)[0]
        else:
            init_feat_cond = self.init_feat_cond_0
            prompt_text_len = 0
            prompt_embed = None
            audio_feat = None

        for sentence in texts:
            if use_text_normalizer:
                sentence = self._text_normalizer.normalize(sentence)

            target_ids = np.array([self.tokenizer(sentence)], dtype=np.int32)
            input_feed_A[self.in_name_A] = ort.OrtValue.ortvalue_from_numpy(target_ids, self.device_type, self.device_id)
            target_embed = self.ort_session_A.run_with_ort_values(self.out_name_A, input_feed_A)[0]

            if use_prompt_audio:
                input_feed_E[self.in_name_E[0]] = prompt_embed
                input_feed_E[self.in_name_E[1]] = target_embed
                target_embed, _ = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)

            input_feed_E[self.in_name_E[0]] = target_embed
            input_feed_E[self.in_name_E[1]] = self.audio_start_embed
            concat_embed, concat_text_len = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)

            if use_prompt_audio:
                input_feed_C[self.in_name_C] = audio_feat
                feat_embed = self.ort_session_C.run_with_ort_values(self.out_name_C, input_feed_C)[0]

                input_feed_E[self.in_name_E[0]] = concat_embed
                input_feed_E[self.in_name_E[1]] = feat_embed
                concat_embed, ids_len = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)
            else:
                feat_embed = self.init_feat_embed
                ids_len = concat_text_len

            concat_text_len_val = int(concat_text_len.numpy().item())
            ids_len_val = int(ids_len.numpy().item())
            max_len = min((concat_text_len_val - prompt_text_len) * self.decode_limit_factor + 10, self.generate_limit - ids_len_val)
            if max_len <= 0:
                continue

            input_feed_F[self.in_name_F[self.num_keys_values]] = self.init_history_len
            input_feed_F[self.in_name_F[self.num_keys_values_plus_1]] = feat_embed
            input_feed_F[self.in_name_F[self.num_keys_values_plus_2]] = concat_text_len
            input_feed_F[self.in_name_F[self.num_keys_values_plus_3]] = concat_embed
            input_feed_F[self.in_name_F[self.num_keys_values_plus_4]] = ids_len
            input_feed_F[self.in_name_F[self.num_keys_values_plus_5]] = self.init_attention_mask_1

            for i in range(self.num_layers):
                input_feed_F[self.in_name_F[i]] = self.init_past_keys_F
            for i in range(self.num_layers, self.num_keys_values):
                input_feed_F[self.in_name_F[i]] = self.init_past_values_F

            feat_cond = init_feat_cond
            
            # Buffering for streaming - accumulate latents before decoding
            latent_buffer = []
            pre_latent_pred = None

            num_decode = 0
            while num_decode < max_len:
                all_outputs_F = self.ort_session_F.run_with_ort_values(self.out_name_F, input_feed_F)

                input_feed_G[self.in_name_G[0]] = init_cfm_steps
                input_feed_G[self.in_name_G[1]] = all_outputs_F[self.num_keys_values_plus_1]
                input_feed_G[self.in_name_G[2]] = all_outputs_F[self.num_keys_values_plus_2]
                input_feed_G[self.in_name_G[3]] = feat_cond

                for _ in range(timesteps):
                    all_outputs_G = self.ort_session_G.run_with_ort_values(self.out_name_G, input_feed_G)
                    input_feed_G[self.in_name_G[0]] = all_outputs_G[0]
                    input_feed_G[self.in_name_G[1]] = all_outputs_G[1]

                latent_pred = all_outputs_G[1]
                latent_buffer.append(latent_pred)

                # Yield audio when we have accumulated enough tokens
                if len(latent_buffer) >= chunk_tokens and pre_latent_pred is not None:
                    # Decode accumulated latents
                    for idx, lat in enumerate(latent_buffer):
                        input_feed_E[self.in_name_E[0]] = pre_latent_pred
                        input_feed_E[self.in_name_E[1]] = lat
                        concat_lat, _ = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)
                        input_feed_H[self.in_name_H] = concat_lat
                        audio_out, _ = self.ort_session_H.run_with_ort_values(self.out_name_H, input_feed_H)
                        pre_latent_pred = lat
                        audio_np = audio_out.numpy()
                        if num_decode > chunk_tokens:
                            audio_np = audio_np[..., self.half_decode_len:]
                        yield audio_np.astype(np.int16).tobytes()
                    latent_buffer = []
                elif pre_latent_pred is None:
                    pre_latent_pred = latent_pred
                    latent_buffer = []

                if num_decode >= self.min_seq_len:
                    stop_id = int(all_outputs_F[self.num_keys_values_plus_3].numpy().item())
                    if stop_id in voxcpm_infer.STOP_TOKEN:
                        break

                input_feed_C[self.in_name_C] = latent_pred
                feat_embed = self.ort_session_C.run_with_ort_values(self.out_name_C, input_feed_C)[0]

                input_feed_D[self.in_name_D] = latent_pred
                feat_cond = self.ort_session_D.run_with_ort_values(self.out_name_D, input_feed_D)[0]

                input_feed_F.update(zip(self.in_name_F[:self.num_keys_values_plus_1], all_outputs_F))
                input_feed_F[self.in_name_F[self.num_keys_values_plus_1]] = feat_embed
                input_feed_F[self.in_name_F[self.num_keys_values_plus_3]] = feat_embed

                if num_decode < 1:
                    input_feed_F[self.in_name_F[self.num_keys_values_plus_2]] = self.init_concat_text_len
                    input_feed_F[self.in_name_F[self.num_keys_values_plus_4]] = self.init_ids_len_1
                    input_feed_F[self.in_name_F[self.num_keys_values_plus_5]] = self.init_attention_mask_0

                num_decode += 1

            # Flush remaining latents
            if latent_buffer and pre_latent_pred is not None:
                for lat in latent_buffer:
                    input_feed_E[self.in_name_E[0]] = pre_latent_pred
                    input_feed_E[self.in_name_E[1]] = lat
                    concat_lat, _ = self.ort_session_E.run_with_ort_values(self.out_name_E, input_feed_E)
                    input_feed_H[self.in_name_H] = concat_lat
                    audio_out, _ = self.ort_session_H.run_with_ort_values(self.out_name_H, input_feed_H)
                    pre_latent_pred = lat
                    audio_np = audio_out.numpy()[..., self.half_decode_len:]
                    yield audio_np.astype(np.int16).tobytes()

            # Yield blank segment between sentences
            yield blank_segment.astype(np.int16).tobytes()

    # ============================================================================
    # Advanced Streaming Features
    # ============================================================================

    def build_prompt_cache(
        self,
        prompt_audio: str,
        prompt_text: str,
        use_text_normalizer: bool | None = None,
        use_audio_normalizer: bool | None = None,
    ) -> PromptCache:
        """Build a prompt cache from audio and text for multi-sentence generation.
        
        This enables maintaining voice consistency across multiple generation calls,
        similar to the official VoxCPM API's build_prompt_cache.
        
        Args:
            prompt_audio: Path to the prompt audio file
            prompt_text: Text content of the prompt audio
            use_text_normalizer: Whether to normalize text (default: engine setting)
            use_audio_normalizer: Whether to normalize audio (default: engine setting)
        
        Returns:
            PromptCache object that can be used with synthesize_with_cache_stream
        """
        use_text_normalizer = self.default_text_normalizer if use_text_normalizer is None else use_text_normalizer
        use_audio_normalizer = self.default_audio_normalizer if use_audio_normalizer is None else use_audio_normalizer

        if use_text_normalizer and self._text_normalizer is None:
            raise RuntimeError("TextNormalizer unavailable")

        voxcpm_infer.ensure_paths(self.models_dir, self.voxcpm_dir, prompt_audio)

        # Read and encode audio
        audio = voxcpm_infer.read_audio_mono_int16(prompt_audio, self.in_sample_rate, self.max_prompt_audio_len)
        if use_audio_normalizer:
            audio = voxcpm_infer.audio_normalizer(audio)
        audio_ort = ort.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), self.device_type, self.device_id)

        input_feed_B = {self.in_name_B: audio_ort}
        audio_feat = self.ort_session_B.run_with_ort_values(self.out_name_B, input_feed_B)[0]

        input_feed_D = {self.in_name_D: audio_feat}
        feat_cond = self.ort_session_D.run_with_ort_values(self.out_name_D, input_feed_D)[0]

        # Tokenize and embed prompt text
        if use_text_normalizer and prompt_text:
            prompt_text = self._text_normalizer.normalize(prompt_text)

        prompt_ids = np.array([self.tokenizer(prompt_text)], dtype=np.int32)
        prompt_text_len = int(prompt_ids.shape[-1])

        input_feed_A = {self.in_name_A: ort.OrtValue.ortvalue_from_numpy(prompt_ids, self.device_type, self.device_id)}
        prompt_embed = self.ort_session_A.run_with_ort_values(self.out_name_A, input_feed_A)[0]

        return PromptCache(
            audio_feat=audio_feat,
            feat_cond=feat_cond,
            prompt_embed=prompt_embed,
            prompt_text_len=prompt_text_len,
            generated_audio_feats=[],
            use_prompt_audio=True,
        )

    def merge_prompt_cache(
        self,
        original_cache: PromptCache,
        new_audio_feats: list,
        max_cache_size: int = 100,
    ) -> PromptCache:
        """Merge newly generated audio features into the prompt cache.
        
        This maintains voice consistency for long/infinite streaming by keeping
        recent generation context in the cache.
        
        Args:
            original_cache: The existing PromptCache
            new_audio_feats: List of newly generated latent features (OrtValues)
            max_cache_size: Maximum number of features to keep (to avoid context overflow)
        
        Returns:
            Updated PromptCache with merged features
        """
        # Combine existing and new features
        combined_feats = original_cache.generated_audio_feats + new_audio_feats
        
        # Trim to max size (keep most recent)
        if len(combined_feats) > max_cache_size:
            combined_feats = combined_feats[-max_cache_size:]
        
        return PromptCache(
            audio_feat=original_cache.audio_feat,
            feat_cond=original_cache.feat_cond,
            prompt_embed=original_cache.prompt_embed,
            prompt_text_len=original_cache.prompt_text_len,
            generated_audio_feats=combined_feats,
            use_prompt_audio=original_cache.use_prompt_audio,
        )

    def _estimate_text_duration(self, text: str) -> float:
        """Estimate expected audio duration in seconds for given text.
        
        Based on typical speech rates:
        - Chinese: ~3-4 characters per second
        - English: ~2-3 words per second
        """
        # Count Chinese characters
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        # Count English words (rough estimate)
        english_words = len([w for w in text.split() if w.isascii() and w.isalpha()])
        
        # Estimate: 3 chars/sec for Chinese, 2.5 words/sec for English
        estimated_seconds = chinese_chars / 3.0 + english_words / 2.5
        
        # Minimum 1 second
        return max(estimated_seconds, 1.0)

    def synthesize_with_retry(
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
        max_retries: int = 3,
        length_ratio_threshold: float = 6.0,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize with automatic retry for badcase detection.
        
        This implements the retry_badcase logic from official VoxCPM:
        - Detects "unstoppable" cases where generated audio is too long
        - Automatically retries with different seeds
        
        Args:
            texts: List of texts to synthesize
            voice: Preset voice name
            prompt_audio: Path to prompt audio
            prompt_text: Text for prompt audio
            voices_file: Path to voices.json
            cfg_value: CFG guidance value
            fixed_timesteps: Number of decoding timesteps
            seed: Random seed (will be incremented on retry)
            use_text_normalizer: Whether to normalize text
            use_audio_normalizer: Whether to normalize audio
            max_retries: Maximum number of retry attempts (default: 3)
            length_ratio_threshold: Max ratio of actual/expected duration (default: 6.0)
        
        Returns:
            Tuple of (audio_array, sample_rate)
        
        Raises:
            RuntimeError: If all retries fail
        """
        seed = int(seed if seed is not None else self.random_seed)
        
        # Calculate expected duration for all texts
        total_text = " ".join(texts)
        expected_duration = self._estimate_text_duration(total_text)
        
        last_audio = None
        last_sr = None
        
        for attempt in range(max_retries):
            current_seed = seed + attempt
            
            try:
                start_time = time.time()
                audio, sr = self.synthesize(
                    texts=texts,
                    voice=voice,
                    prompt_audio=prompt_audio,
                    prompt_text=prompt_text,
                    voices_file=voices_file,
                    cfg_value=cfg_value,
                    fixed_timesteps=fixed_timesteps,
                    seed=current_seed,
                    use_text_normalizer=use_text_normalizer,
                    use_audio_normalizer=use_audio_normalizer,
                )
                elapsed = time.time() - start_time
                
                last_audio = audio
                last_sr = sr
                
                # Calculate actual duration
                actual_duration = len(audio) / sr
                duration_ratio = actual_duration / expected_duration
                
                # Check for badcase
                if duration_ratio <= length_ratio_threshold:
                    # Good case - return
                    if attempt > 0:
                        print(f"Retry #{attempt} succeeded (duration ratio: {duration_ratio:.2f})")
                    return audio, sr
                
                print(f"Badcase detected (attempt {attempt + 1}/{max_retries}): "
                      f"duration ratio {duration_ratio:.2f} > threshold {length_ratio_threshold}")
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
        
        # All retries exhausted, return last result with warning
        print(f"Warning: All {max_retries} retries exhausted, returning last result")
        if last_audio is None:
            raise RuntimeError("All synthesis attempts failed")
        
        return last_audio, last_sr
