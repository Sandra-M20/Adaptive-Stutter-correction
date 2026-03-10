"""
preprocessing.py
================
Pipeline Steps 1 & 2: Audio Input and Preprocessing

This module handles all audio I/O and signal preparation:

Step 1 — Audio Input:
  - Accept a file path (WAV, MP3, FLAC, OGG) or a numpy (signal, sr) tuple
  - Read audio using soundfile (stable, no DLL issues)
  - Convert stereo/multi-channel to mono

Step 2 — Audio Preprocessing:
  - Resample to TARGET_SR (22050 Hz) using linear interpolation
  - Apply noise reduction (Spectral Subtraction or Wiener filter)
  - Normalize amplitude to [-1.0, 1.0]
  - Trim leading and trailing silence

All operations are pure numpy / soundfile — no librosa, no scipy.signal.
"""

import os
import numpy as np
import soundfile as sf

from config import TARGET_SR
from utils import resample, normalize
from noise_reduction import NoiseReducer


class AudioPreprocessor:
    """
    Steps 1 & 2: Load, resample, denoise, and normalize audio.

    Parameters
    ----------
    target_sr     : int  — output sample rate (default: 22050 Hz)
    noise_reduce  : bool — apply noise reduction (default: True)
    nr_method     : str  — 'spectral' or 'wiener'
    trim_silence  : bool — strip leading/trailing silence (default: True)
    trim_db       : float— silence threshold in dB for trimming (default: -50)
    """

    def __init__(self,
                 target_sr: int    = TARGET_SR,
                 noise_reduce: bool = True,
                 nr_method: str    = "spectral",
                 trim_silence: bool = True,
                 trim_db: float    = -50.0):
        self.target_sr    = target_sr
        self.noise_reduce = noise_reduce
        self.nr_method    = nr_method
        self.trim_silence = trim_silence
        self.trim_db      = trim_db
        self._reducer     = None  # Lazy-init per call (sr may vary)

    # ------------------------------------------------------------------ #

    def process(self, audio_input) -> tuple:
        """
        Full preprocessing pipeline (Steps 1-2).

        Parameters
        ----------
        audio_input : str | tuple(np.ndarray, int)
            - str   : file path to audio file
            - tuple : (signal_array, sample_rate)

        Returns
        -------
        signal : np.ndarray — preprocessed mono float32 signal
        sr     : int        — sample rate (= self.target_sr)
        """
        # Step 1: Load
        signal, sr = self._load(audio_input)
        if signal.size == 0:
            raise ValueError("Audio signal is empty.")
        if not np.isfinite(signal).all():
            raise ValueError("Audio contains invalid samples (NaN or Inf).")
        print(f"[Preprocessing] Loaded: {len(signal)/sr:.2f}s | "
              f"SR={sr}Hz | Channels=1 (mono)")

        # Resample
        if sr != self.target_sr:
            signal = resample(signal, sr, self.target_sr)
            sr     = self.target_sr
            print(f"[Preprocessing] Resampled to {sr}Hz.")

        # Remove DC offset and very-low-frequency rumble before denoising.
        signal = self._dc_block(signal)

        # Step 2a: Noise reduction
        if self.noise_reduce:
            reducer = NoiseReducer(method=self.nr_method)
            signal  = reducer.process(signal)

        # Step 2b: Trim silence
        if self.trim_silence:
            signal = self._trim(signal, sr)

        # Step 2c: Normalize
        signal = normalize(signal)
        signal = np.clip(signal, -1.0, 1.0)
        print(f"[Preprocessing] Final: {len(signal)/sr:.3f}s | "
              f"Peak={np.max(np.abs(signal)):.4f}")

        return signal, sr

    # ------------------------------------------------------------------ #

    def _load(self, audio_input) -> tuple:
        """Load audio from file path or (array, sr) tuple."""
        if isinstance(audio_input, (str, os.PathLike)):
            signal, sr = sf.read(str(audio_input), dtype="float32", always_2d=False)
            if signal.ndim == 2:
                signal = signal.mean(axis=1)   # stereo -> mono
        elif isinstance(audio_input, tuple):
            signal, sr = audio_input
            signal = signal.astype(np.float32)
            if signal.ndim == 2:
                signal = signal.mean(axis=1)
        else:
            raise TypeError(f"audio_input must be path or (array, sr) tuple, "
                            f"got {type(audio_input)}")
        return signal, int(sr)

    def _trim(self, signal: np.ndarray, sr: int,
              frame_ms: int = 20) -> np.ndarray:
        """
        Remove leading and trailing silence below `trim_db` dBFS.
        Uses short-time energy framing.
        """
        frame_size = int(sr * frame_ms / 1000)
        threshold  = 10 ** (self.trim_db / 20.0)   # convert dB to amplitude
        energies   = [
            np.sqrt(np.mean(signal[s:s + frame_size] ** 2))
            for s in range(0, len(signal) - frame_size + 1, frame_size)
        ]
        active = [i for i, e in enumerate(energies) if e > threshold]
        if not active:
            return signal
        start = active[0] * frame_size
        end   = min((active[-1] + 1) * frame_size, len(signal))
        trimmed = signal[start:end]
        removed = (len(signal) - len(trimmed)) / sr
        if removed > 0.01:
            print(f"[Preprocessing] Trimmed {removed:.3f}s of silence "
                  f"({start} leading + {len(signal)-end} trailing samples).")
        return trimmed

    def _dc_block(self, signal: np.ndarray, hp_hz: float = 40.0) -> np.ndarray:
        """
        One-pole DC blocker / high-pass (pure numpy, sample-by-sample IIR).
        """
        x = signal.astype(np.float32)
        if len(x) < 2:
            return x
        # Pole near 1.0; mapped from desired high-pass corner.
        r = float(np.exp(-2.0 * np.pi * hp_hz / max(self.target_sr, 1)))
        y = np.zeros_like(x, dtype=np.float32)
        y[0] = x[0]
        for n in range(1, len(x)):
            y[n] = x[n] - x[n - 1] + r * y[n - 1]
        return y

    # ------------------------------------------------------------------ #

    def get_info(self, file_path: str) -> dict:
        """
        Return metadata about an audio file without fully loading it.
        """
        info = sf.info(file_path)
        return {
            "path":      file_path,
            "duration":  info.duration,
            "sr":        info.samplerate,
            "channels":  info.channels,
            "format":    info.format,
            "subtype":   info.subtype,
        }
