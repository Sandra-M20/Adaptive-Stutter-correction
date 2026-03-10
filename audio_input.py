"""
audio_input.py
==============
Audio input utilities for the Adaptive Enhancement of Stuttered Speech
Correction system.

Supports:
1. Loading audio from file (.wav/.flac/.ogg/.mp3 if backend supports)
2. Capturing short microphone recordings (optional; requires sounddevice)
"""

from __future__ import annotations

from typing import Tuple
import os
import numpy as np
import soundfile as sf

from utils import resample, normalize


class AudioInputManager:
    """Load or capture audio and return mono float32 waveform + sample rate."""

    def __init__(self, target_sr: int = 22050):
        self.target_sr = int(target_sr)

    def from_file(self, path: str, normalize_audio: bool = True) -> Tuple[np.ndarray, int]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        sig, sr = sf.read(path, dtype="float32", always_2d=False)
        if sig.ndim == 2:
            sig = sig.mean(axis=1)
        if int(sr) != self.target_sr:
            sig = resample(sig, int(sr), self.target_sr)
            sr = self.target_sr
        sig = sig.astype(np.float32)
        if sig.size == 0:
            raise ValueError("Audio is empty")
        if len(sig) < int(0.1 * sr):
            raise ValueError("Audio too short")
        if not np.isfinite(sig).all():
            raise ValueError("Audio contains invalid samples (NaN/Inf)")
        if np.max(np.abs(sig)) < 1e-6:
            raise ValueError("Audio is silent")
        if normalize_audio:
            sig = normalize(sig)
        sig = np.clip(sig, -1.0, 1.0).astype(np.float32)
        return sig, int(sr)

    def from_microphone(self, duration_s: float = 5.0, normalize_audio: bool = True) -> Tuple[np.ndarray, int]:
        """
        Record mono audio using default microphone.
        Requires: `pip install sounddevice`.
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise RuntimeError("Microphone capture requires `sounddevice` package.") from e

        n = int(max(duration_s, 0.1) * self.target_sr)
        audio = sd.rec(n, samplerate=self.target_sr, channels=1, dtype="float32")
        sd.wait()
        sig = audio[:, 0].astype(np.float32)
        if sig.size == 0:
            raise ValueError("Recorded audio is empty")
        if len(sig) < int(0.1 * self.target_sr):
            raise ValueError("Audio too short")
        if not np.isfinite(sig).all():
            raise ValueError("Recorded audio contains invalid samples (NaN/Inf)")
        if np.max(np.abs(sig)) < 1e-6:
            raise ValueError("Audio is silent")
        if normalize_audio:
            sig = normalize(sig)
        sig = np.clip(sig, -1.0, 1.0).astype(np.float32)
        return sig, self.target_sr
