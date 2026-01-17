"""Abstract base class for VoxCPM TTS engines."""

from abc import ABC, abstractmethod
from typing import Generator, List

import numpy as np


class BaseEngine(ABC):
    """Abstract base class for VoxCPM TTS engines.
    
    All engine implementations (0.5B, 1.5B) must inherit from this class
    and implement the required abstract methods.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output audio sample rate in Hz."""
        pass

    @property
    @abstractmethod
    def bit_depth(self) -> int:
        """Output audio bit depth (typically 16)."""
        pass

    @property
    @abstractmethod
    def channels(self) -> int:
        """Number of audio channels (typically 1 for mono)."""
        pass

    def get_audio_info(self) -> dict:
        """Get audio format information for client configuration.
        
        Returns:
            Dict with sample_rate, bit_depth, and channels
        """
        return {
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
        }

    @abstractmethod
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
        """Synthesize speech from text.
        
        Args:
            texts: List of text strings to synthesize
            voice: Name of voice preset from voices.json
            prompt_audio: Path to reference audio for voice cloning
            prompt_text: Transcript of the reference audio
            voices_file: Path to voices.json (optional override)
            cfg_value: CFG guidance value
            fixed_timesteps: Number of diffusion timesteps
            seed: Random seed for reproducibility
            streaming: Whether to use streaming decode mode
            use_text_normalizer: Enable text normalization
            use_audio_normalizer: Enable audio normalization
        
        Returns:
            Tuple of (audio_array, sample_rate)
            - audio_array: np.ndarray of int16 audio samples
            - sample_rate: Sample rate in Hz
        
        Raises:
            ValueError: If voice and prompt_audio/prompt_text are both provided
            FileNotFoundError: If required files are missing
        """
        pass

    @abstractmethod
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
        """Synthesize speech from text with streaming output.
        
        Yields raw PCM audio chunks (int16) as they are generated.
        This enables low-latency playback where audio plays while
        generation is still in progress.
        
        Args:
            texts: List of text strings to synthesize
            voice: Name of voice preset from voices.json
            prompt_audio: Path to reference audio for voice cloning
            prompt_text: Transcript of the reference audio
            voices_file: Path to voices.json (optional override)
            cfg_value: CFG guidance value
            fixed_timesteps: Number of diffusion timesteps
            seed: Random seed for reproducibility
            use_text_normalizer: Enable text normalization
            use_audio_normalizer: Enable audio normalization
            chunk_tokens: Number of tokens to accumulate before yielding
                         (default 4, for ~50ms chunks)
        
        Yields:
            Raw PCM bytes (int16, mono) ready for playback
        
        Raises:
            ValueError: If voice and prompt_audio/prompt_text are both provided
            FileNotFoundError: If required files are missing
        """
        pass
