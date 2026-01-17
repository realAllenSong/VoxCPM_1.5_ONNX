from __future__ import annotations

import asyncio
import json
import os
import queue
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import LlamaTokenizerFast

import infer as voxcpm_infer


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_text_value(value: str | List[str]) -> List[str]:
    if isinstance(value, list):
        texts = [str(item).strip() for item in value if str(item).strip()]
        return texts
    if not isinstance(value, str):
        raise ValueError("text must be a string or list of strings")
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    if "\n" in text:
        return [line.strip() for line in text.splitlines() if line.strip()]
    return [text]


class SynthesisRequest(BaseModel):
    text: str | List[str]
    voice: str | None = None
    prompt_text: str | None = None
    prompt_audio_path: str | None = None
    voices_file: str | None = None
    cfg_value: float | None = None
    fixed_timesteps: int | None = None
    seed: int | None = None
    text_normalizer: bool | None = None
    audio_normalizer: bool | None = None
    chunk_tokens: int | None = None  # For streaming: tokens per chunk
    # Retry mechanism for badcase handling
    retry_badcase: bool = False  # Enable automatic retry for unstoppable cases
    max_retries: int = 3  # Maximum retry attempts
    length_ratio_threshold: float = 6.0  # Max ratio of actual/expected duration


class VoxCPMEngine:
    def __init__(
        self,
        models_dir: str,
        voxcpm_dir: str,
        voices_file: str,
        onnx_config: str | None,
        max_threads: int,
        text_normalizer: bool,
        audio_normalizer: bool,
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

        self.text_normalizer = None
        if self.default_text_normalizer:
            if voxcpm_infer.TextNormalizer is None:
                raise RuntimeError(
                    "TextNormalizer unavailable. Install dependencies: pip install wetext regex inflect "
                    f"(error: {voxcpm_infer.TEXT_NORMALIZER_IMPORT_ERROR})"
                )
            self.text_normalizer = voxcpm_infer.TextNormalizer()

        self.device_type = "cpu"
        self.device_id = 0

        self._init_sessions(max_threads)
        self.tokenizer = voxcpm_infer.mask_multichar_chinese_tokens(
            LlamaTokenizerFast.from_pretrained(self.voxcpm_dir)
        )

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
        voice: str | None,
        prompt_audio: str | None,
        prompt_text: str | None,
        voices_file: str | None,
        cfg_value: float | None,
        fixed_timesteps: int | None,
        seed: int | None,
        streaming: bool,
        use_text_normalizer: bool | None,
        use_audio_normalizer: bool | None,
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
        if use_text_normalizer and self.text_normalizer is None:
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
                prompt_text = self.text_normalizer.normalize(prompt_text)

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
                sentence = self.text_normalizer.normalize(sentence)

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


def build_engine_queue() -> tuple[queue.Queue, ThreadPoolExecutor]:
    models_dir = os.getenv("VOXCPM_MODELS_DIR", os.path.join(os.getcwd(), "models", "onnx_models"))
    voxcpm_dir = os.getenv("VOXCPM_VOXCPM_DIR", os.path.join(os.getcwd(), "models", "VoxCPM1.5"))
    voices_file = os.getenv("VOXCPM_VOICES_FILE", os.path.join(os.getcwd(), "voices.json"))
    onnx_config = os.getenv("VOXCPM_ONNX_CONFIG")
    max_threads = int(os.getenv("VOXCPM_MAX_THREADS", "0"))
    text_normalizer = os.getenv("VOXCPM_TEXT_NORMALIZER", "true").lower() in {"1", "true", "yes", "y"}
    audio_normalizer = os.getenv("VOXCPM_AUDIO_NORMALIZER", "false").lower() in {"1", "true", "yes", "y"}
    max_concurrency = int(os.getenv("VOXCPM_MAX_CONCURRENCY", "1"))

    engine_queue = queue.Queue()
    for _ in range(max_concurrency):
        engine_queue.put(
            VoxCPMEngine(
                models_dir=models_dir,
                voxcpm_dir=voxcpm_dir,
                voices_file=voices_file,
                onnx_config=onnx_config,
                max_threads=max_threads,
                text_normalizer=text_normalizer,
                audio_normalizer=audio_normalizer,
            )
        )

    executor = ThreadPoolExecutor(max_workers=max_concurrency)
    return engine_queue, executor


ENGINE_QUEUE: queue.Queue[VoxCPMEngine]
EXECUTOR: ThreadPoolExecutor

app = FastAPI(title="VoxCPM ONNX API", version="1.0")


@app.on_event("startup")
def on_startup() -> None:
    global ENGINE_QUEUE, EXECUTOR
    ENGINE_QUEUE, EXECUTOR = build_engine_queue()


@app.on_event("shutdown")
def on_shutdown() -> None:
    EXECUTOR.shutdown(wait=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/voices")
def list_voices(voices_file: str | None = None) -> dict:
    target = voices_file or os.getenv("VOXCPM_VOICES_FILE") or os.path.join(os.getcwd(), "voices.json")
    voices = voxcpm_infer.load_voice_presets(target)
    return {"voices": sorted(voices.keys())}


def synthesize_to_file(
    request_texts: List[str],
    voice: str | None,
    prompt_audio: str | None,
    prompt_text: str | None,
    voices_file: str | None,
    cfg_value: float | None,
    fixed_timesteps: int | None,
    seed: int | None,
    text_normalizer: bool | None,
    audio_normalizer: bool | None,
    output_path: str,
) -> str:
    engine = ENGINE_QUEUE.get()
    try:
        audio_out, sample_rate = engine.synthesize(
            texts=request_texts,
            voice=voice,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            voices_file=voices_file,
            cfg_value=cfg_value,
            fixed_timesteps=fixed_timesteps,
            seed=seed,
            streaming=False,
            use_text_normalizer=text_normalizer,
            use_audio_normalizer=audio_normalizer,
        )
        sf.write(output_path, audio_out, sample_rate, format="WAV", subtype="PCM_16")
        return output_path
    finally:
        ENGINE_QUEUE.put(engine)


@app.post("/synthesize")
async def synthesize_json(request: SynthesisRequest, background_tasks: BackgroundTasks):
    texts = parse_text_value(request.text)
    if not texts:
        raise HTTPException(status_code=400, detail="text is required")

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = output_file.name
    output_file.close()

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            EXECUTOR,
            synthesize_to_file,
            texts,
            request.voice,
            request.prompt_audio_path,
            request.prompt_text,
            request.voices_file,
            request.cfg_value,
            request.fixed_timesteps,
            request.seed,
            request.text_normalizer,
            request.audio_normalizer,
            output_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(os.remove, output_path)
    return FileResponse(output_path, media_type="audio/wav", filename="output.wav", background=background_tasks)


@app.post("/synthesize-file")
async def synthesize_file(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    voice: str | None = Form(None),
    prompt_text: str | None = Form(None),
    voices_file: str | None = Form(None),
    cfg_value: float | None = Form(None),
    fixed_timesteps: int | None = Form(None),
    seed: int | None = Form(None),
    text_normalizer: str | None = Form(None),
    audio_normalizer: str | None = Form(None),
    prompt_audio: UploadFile | None = File(None),
):
    try:
        texts = parse_text_value(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prompt_audio_path = None
    if prompt_audio is not None:
        suffix = os.path.splitext(prompt_audio.filename or "")[1] or ".wav"
        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        prompt_audio_path = tmp_audio.name
        tmp_audio.write(await prompt_audio.read())
        tmp_audio.close()
        background_tasks.add_task(os.remove, prompt_audio_path)

    try:
        text_normalizer_value = parse_bool(text_normalizer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        audio_normalizer_value = parse_bool(audio_normalizer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = output_file.name
    output_file.close()

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            EXECUTOR,
            synthesize_to_file,
            texts,
            voice,
            prompt_audio_path,
            prompt_text,
            voices_file,
            cfg_value,
            fixed_timesteps,
            seed,
            text_normalizer_value,
            audio_normalizer_value,
            output_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(os.remove, output_path)
    return FileResponse(output_path, media_type="audio/wav", filename="output.wav", background=background_tasks)


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/info")
def audio_info() -> dict:
    """Get audio format information for client configuration.
    
    Returns sample rate, bit depth, and channels for configuring
    audio playback (e.g., PyAudio).
    """
    # Peek at an engine without removing from queue
    engine = ENGINE_QUEUE.queue[0]
    return {
        "sample_rate": engine.out_sample_rate,
        "bit_depth": 16,
        "channels": 1,
        "format": "int16",
    }


def stream_generator(
    texts: List[str],
    voice: str | None,
    prompt_audio: str | None,
    prompt_text: str | None,
    voices_file: str | None,
    cfg_value: float | None,
    fixed_timesteps: int | None,
    seed: int | None,
    text_normalizer: bool | None,
    audio_normalizer: bool | None,
    chunk_tokens: int,
):
    """Generator that yields audio chunks from the engine."""
    engine = ENGINE_QUEUE.get()
    try:
        for chunk in engine.synthesize(
            texts=texts,
            voice=voice,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            voices_file=voices_file,
            cfg_value=cfg_value,
            fixed_timesteps=fixed_timesteps,
            seed=seed,
            streaming=True,
            use_text_normalizer=text_normalizer,
            use_audio_normalizer=audio_normalizer,
        ):
            # For now, use non-streaming and yield the full result
            # TODO: Implement proper streaming in VoxCPMEngine.synthesize
            pass
    except Exception:
        ENGINE_QUEUE.put(engine)
        raise
    ENGINE_QUEUE.put(engine)
    
    # Fallback: Use synchronous synthesis and yield result as single chunk
    engine = ENGINE_QUEUE.get()
    try:
        audio_out, sample_rate = engine.synthesize(
            texts=texts,
            voice=voice,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            voices_file=voices_file,
            cfg_value=cfg_value,
            fixed_timesteps=fixed_timesteps,
            seed=seed,
            streaming=False,
            use_text_normalizer=text_normalizer,
            use_audio_normalizer=audio_normalizer,
        )
        yield audio_out.astype(np.int16).tobytes()
    finally:
        ENGINE_QUEUE.put(engine)


@app.post("/synthesize-stream")
async def synthesize_stream(request: SynthesisRequest):
    """Stream raw PCM audio as it is generated.
    
    Returns raw PCM bytes (int16, mono) suitable for real-time playback.
    Use the /info endpoint to get audio format parameters.
    
    Headers:
        X-Sample-Rate: Audio sample rate (e.g., 44100 for 1.5B, 16000 for 0.5B)
        X-Bit-Depth: Bits per sample (16)
        X-Channels: Number of channels (1)
    """
    texts = parse_text_value(request.text)
    if not texts:
        raise HTTPException(status_code=400, detail="text is required")

    chunk_tokens = request.chunk_tokens or 4
    
    # Get audio info for headers
    engine = ENGINE_QUEUE.queue[0]
    sample_rate = engine.out_sample_rate

    def generate():
        eng = ENGINE_QUEUE.get()
        try:
            audio_out, _ = eng.synthesize(
                texts=texts,
                voice=request.voice,
                prompt_audio=request.prompt_audio_path,
                prompt_text=request.prompt_text,
                voices_file=request.voices_file,
                cfg_value=request.cfg_value,
                fixed_timesteps=request.fixed_timesteps,
                seed=request.seed,
                streaming=False,
                use_text_normalizer=request.text_normalizer,
                use_audio_normalizer=request.audio_normalizer,
            )
            # Yield audio in chunks for streaming response
            chunk_size = sample_rate // 10  # ~100ms chunks
            audio_bytes = audio_out.astype(np.int16).tobytes()
            for i in range(0, len(audio_bytes), chunk_size * 2):  # *2 for int16
                yield audio_bytes[i:i + chunk_size * 2]
        finally:
            ENGINE_QUEUE.put(eng)

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Bit-Depth": "16",
            "X-Channels": "1",
            "Content-Type": "application/octet-stream",
        }
    )
