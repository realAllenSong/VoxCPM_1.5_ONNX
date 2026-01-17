"""VoxCPM TTS Engines package.

This package provides abstraction for different VoxCPM model sizes (0.5B, 1.5B).
Each engine implements the same interface but uses different ONNX model architectures.
"""

from .base import BaseEngine
from .voxcpm_15b import VoxCPM15BEngine

__all__ = ["BaseEngine", "VoxCPM15BEngine", "load_engine"]


def load_engine(
    model_size: str,
    models_dir: str,
    voxcpm_dir: str,
    voices_file: str,
    onnx_config: str | None = None,
    max_threads: int = 0,
    text_normalizer: bool = True,
    audio_normalizer: bool = False,
) -> BaseEngine:
    """Factory function to load the appropriate engine based on model size.
    
    Args:
        model_size: Model size to use ("0.5b" or "1.5b")
        models_dir: Directory containing ONNX model files
        voxcpm_dir: Directory containing tokenizer files
        voices_file: Path to voices.json preset file
        onnx_config: Optional path to ONNX config JSON
        max_threads: Max CPU threads (0 = auto)
        text_normalizer: Whether to enable text normalization
        audio_normalizer: Whether to enable audio normalization
    
    Returns:
        Initialized engine instance
    
    Raises:
        ValueError: If model_size is not supported
    """
    model_size = model_size.lower().strip()
    
    if model_size in ("0.5b", "05b", "0.5"):
        # Import here to avoid circular imports and allow lazy loading
        from .voxcpm_05b import VoxCPM05BEngine
        return VoxCPM05BEngine(
            models_dir=models_dir,
            voxcpm_dir=voxcpm_dir,
            voices_file=voices_file,
            onnx_config=onnx_config,
            max_threads=max_threads,
            text_normalizer=text_normalizer,
            audio_normalizer=audio_normalizer,
        )
    elif model_size in ("1.5b", "15b", "1.5"):
        return VoxCPM15BEngine(
            models_dir=models_dir,
            voxcpm_dir=voxcpm_dir,
            voices_file=voices_file,
            onnx_config=onnx_config,
            max_threads=max_threads,
            text_normalizer=text_normalizer,
            audio_normalizer=audio_normalizer,
        )
    else:
        raise ValueError(
            f"Unsupported model size: {model_size}. "
            "Supported sizes: '0.5b', '1.5b'"
        )
