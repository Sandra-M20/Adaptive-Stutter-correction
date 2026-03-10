"""
noise_reduction.py
==================
Advanced Audio Noise Reduction Module

Provides two complementary noise reduction strategies:
  1. SpectralSubtractor  — classic spectral subtraction (low latency)
  2. WienerEstimator     — MMSE-based Wiener filter (higher quality)

Both use only numpy. No scipy.signal calls (avoids Windows DLL crash).

Reference:
  Boll, S.F. (1979). "Suppression of Acoustic Noise in Speech Using
  Spectral Subtraction." IEEE TASP 27(2):113-120.
"""

import numpy as np
from config import N_FFT, HOP_STFT, NOISE_FRAMES
from utils import stft, istft


class SpectralSubtractor:
    """
    Spectral Subtraction noise reduction.

    Estimates the noise floor from the first `noise_frames` frames
    (assumed to be silence / background noise) and subtracts the
    estimated noise magnitude from every frame.

    Parameters
    ----------
    n_fft        : FFT size
    hop          : STFT hop size
    noise_frames : Number of leading frames for noise estimation
    alpha        : Over-subtraction factor (2.0 recommended)
    beta         : Spectral floor factor (prevents full nulling)
    """

    def __init__(self, n_fft: int = N_FFT, hop: int = HOP_STFT,
                 noise_frames: int = NOISE_FRAMES,
                 alpha: float = 2.0, beta: float = 0.01):
        self.n_fft        = n_fft
        self.hop          = hop
        self.noise_frames = noise_frames
        self.alpha        = alpha
        self.beta         = beta

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction to `signal`.
        Returns clean signal of same length.
        """
        print("[NoiseReduction] Spectral subtraction started...")
        frames = stft(signal, n_fft=self.n_fft, hop=self.hop)
        if len(frames) == 0:
            return signal

        mag   = np.abs(frames)
        phase = np.angle(frames)

        # Noise profile: estimate from lowest-energy frames (more robust than
        # assuming initial frames are silence).
        frame_energy = np.mean(power := (mag ** 2), axis=1)
        n_pick = min(self.noise_frames, len(frames))
        idx = np.argsort(frame_energy)[:n_pick]
        noise = np.mean(mag[idx], axis=0)

        # Adaptive subtraction strength from rough global SNR.
        noise_pwr = float(np.mean(frame_energy[idx])) + 1e-10
        signal_pwr = float(np.mean(frame_energy)) + 1e-10
        snr_db = 10.0 * np.log10(max(signal_pwr - noise_pwr, 1e-10) / noise_pwr)
        if snr_db >= 22.0:
            alpha = 0.8   # already clean: do very light denoising
        elif snr_db >= 12.0:
            alpha = 1.2
        else:
            alpha = self.alpha

        # Subtraction with spectral floor
        clean_mag = np.maximum(mag - alpha * noise, self.beta * mag)
        clean_stft = clean_mag * np.exp(1j * phase)

        out = istft(clean_stft, hop=self.hop, signal_len=len(signal))
        print(f"[NoiseReduction] Spectral subtraction complete. "
              f"SNR~{snr_db:.1f}dB alpha={alpha:.2f} | "
              f"Signal length preserved: {len(out)} samples.")
        return out.astype(np.float32)


class WienerEstimator:
    """
    Minimum Mean Square Error (MMSE) Wiener Filter.

    Computes a per-bin Wiener gain H(k) = SNR(k) / (1 + SNR(k))
    and applies it to each STFT frame. Produces less musical noise
    than standard spectral subtraction.
    """

    def __init__(self, n_fft: int = N_FFT, hop: int = HOP_STFT,
                 noise_frames: int = NOISE_FRAMES):
        self.n_fft        = n_fft
        self.hop          = hop
        self.noise_frames = noise_frames

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Apply Wiener filtering to `signal`. Returns clean signal."""
        print("[NoiseReduction] Wiener filter started...")
        frames = stft(signal, n_fft=self.n_fft, hop=self.hop)
        if len(frames) == 0:
            return signal

        mag   = np.abs(frames)
        phase = np.angle(frames)
        power = mag ** 2

        # Noise power from lowest-energy frames.
        frame_energy = np.mean(power, axis=1)
        n_pick = min(self.noise_frames, len(frames))
        idx = np.argsort(frame_energy)[:n_pick]
        noise_pwr = np.mean(power[idx], axis=0)
        noise_pwr = np.maximum(noise_pwr, 1e-10)

        # Wiener gain per bin (floored at 0)
        snr  = np.maximum(power - noise_pwr, 0) / noise_pwr
        gain = snr / (1.0 + snr)        # shape: (T, F)

        clean_mag  = gain * mag
        clean_stft = clean_mag * np.exp(1j * phase)
        out = istft(clean_stft, hop=self.hop, signal_len=len(signal))
        print("[NoiseReduction] Wiener filter complete.")
        return out.astype(np.float32)


class NoiseReducer:
    """
    Convenience wrapper that selects between SpectralSubtractor and
    WienerEstimator based on `method` parameter.
    """

    METHODS = {"spectral": SpectralSubtractor, "wiener": WienerEstimator}

    def __init__(self, method: str = "spectral", **kwargs):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {list(self.METHODS)}")
        self._reducer = self.METHODS[method](**kwargs)
        self.method   = method

    def process(self, signal: np.ndarray) -> np.ndarray:
        return self._reducer.process(signal)
