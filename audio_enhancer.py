"""
audio_enhancer.py
=================
DSP post-processing stage for speech clarity.

Goals:
  - Suppress residual background noise without musical artifacts
  - Keep speech band and reduce low/high-frequency clutter
  - Tame sibilant harshness
  - Stabilize loudness while preserving speech dynamics
"""

import numpy as np

from config import TARGET_SR, N_FFT, HOP_STFT
from utils import stft, istft, normalize


class AudioEnhancer:
    """
    Post-correction speech enhancement chain.
    """

    def __init__(
        self,
        sr: int = TARGET_SR,
        target_peak_dbfs: float = -1.0,
        target_rms_dbfs: float = -16.0,
        make_up_gain_db: float = 5.0,
        enable_bandpass: bool = True,
        enable_gate: bool = True,
        enable_de_esser: bool = True,
        enable_compressor: bool = True,
    ):
        self.sr = sr
        self.target_peak_dbfs = target_peak_dbfs
        self.target_rms_dbfs = target_rms_dbfs
        self.make_up_gain_db = make_up_gain_db
        self.enable_bandpass = enable_bandpass
        self.enable_gate = enable_gate
        self.enable_de_esser = enable_de_esser
        self.enable_compressor = enable_compressor

    def _bandpass_spectral(self, signal: np.ndarray, f_lo: float = 70.0, f_hi: float = 7600.0) -> np.ndarray:
        """
        Speech-band emphasis in frequency domain with soft edges.
        """
        X = stft(signal, n_fft=N_FFT, hop=HOP_STFT)
        if len(X) == 0:
            return signal.astype(np.float32)

        n_bins = X.shape[1]
        freqs = np.linspace(0.0, self.sr / 2.0, n_bins)

        # Soft roll-on and roll-off instead of hard cuts.
        lo_w = np.clip((freqs - f_lo) / max(40.0, f_lo * 0.6), 0.0, 1.0)
        hi_w = np.clip((f_hi - freqs) / max(500.0, f_hi * 0.2), 0.0, 1.0)
        w = np.minimum(lo_w, hi_w)
        w = 0.25 + 0.75 * w  # keep some ambience instead of full nulling

        X *= w[np.newaxis, :]
        out = istft(X, hop=HOP_STFT, signal_len=len(signal))
        return out.astype(np.float32)

    def _spectral_gate(self, signal: np.ndarray, noise_frames: int = 24) -> np.ndarray:
        """
        Adaptive soft spectral gate using low-energy frame statistics.
        """
        X = stft(signal, n_fft=N_FFT, hop=HOP_STFT)
        if len(X) == 0:
            return signal.astype(np.float32)

        mag = np.abs(X)
        phase = np.angle(X)
        pwr = mag ** 2
        frame_energy = np.mean(pwr, axis=1)

        n_pick = min(noise_frames, len(frame_energy))
        idx = np.argsort(frame_energy)[:n_pick]
        noise = np.maximum(np.median(mag[idx], axis=0), 1e-8)

        # Soft mask: values near noise floor get attenuated, not removed.
        ratio = mag / noise[np.newaxis, :]
        mask = np.clip((ratio - 1.0) / 2.5, 0.08, 1.0)

        # Temporal smoothing to reduce musical noise.
        for t in range(1, mask.shape[0]):
            mask[t] = 0.65 * mask[t - 1] + 0.35 * mask[t]

        Y = (mag * mask) * np.exp(1j * phase)
        out = istft(Y, hop=HOP_STFT, signal_len=len(signal))
        return out.astype(np.float32)

    def _de_esser(self, signal: np.ndarray, sibilant_lo: float = 4200.0, reduction_db: float = 4.0) -> np.ndarray:
        """
        Frame-adaptive attenuation of high-band sibilance.
        """
        X = stft(signal, n_fft=N_FFT, hop=HOP_STFT)
        if len(X) == 0:
            return signal.astype(np.float32)

        mag = np.abs(X)
        phase = np.angle(X)

        n_bins = X.shape[1]
        freqs = np.linspace(0.0, self.sr / 2.0, n_bins)
        hf_mask = freqs >= sibilant_lo
        lf_mask = (freqs >= 300.0) & (freqs < 3000.0)

        hf_e = np.mean(mag[:, hf_mask] ** 2, axis=1) + 1e-10
        lf_e = np.mean(mag[:, lf_mask] ** 2, axis=1) + 1e-10
        ratio = hf_e / lf_e
        thr = np.percentile(ratio, 75)

        max_red = 10 ** (-reduction_db / 20.0)
        per_frame_gain = np.ones(len(ratio), dtype=np.float32)
        hot = ratio > thr
        per_frame_gain[hot] = np.clip(thr / ratio[hot], max_red, 1.0)

        gain = np.ones_like(mag, dtype=np.float32)
        gain[:, hf_mask] = per_frame_gain[:, np.newaxis]

        Y = (mag * gain) * np.exp(1j * phase)
        out = istft(Y, hop=HOP_STFT, signal_len=len(signal))
        return out.astype(np.float32)

    def _compress(self, signal: np.ndarray, threshold: float = 0.20, ratio: float = 3.2) -> np.ndarray:
        """
        Light feed-forward compressor.
        """
        env = np.maximum(np.abs(signal), 1e-8)
        over = env > threshold
        gain = np.ones_like(signal)
        gain[over] = (threshold + (env[over] - threshold) / ratio) / env[over]

        # Smooth gain to avoid pumping.
        attack = 0.12
        release = 0.03
        g = np.ones_like(gain)
        for i in range(1, len(g)):
            a = attack if gain[i] < g[i - 1] else release
            g[i] = (1 - a) * g[i - 1] + a * gain[i]

        y = signal * g
        # Post-compressor makeup gain increases perceived loudness.
        y *= float(10 ** (self.make_up_gain_db / 20.0))
        return y.astype(np.float32)

    def _rms_normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Loudness-oriented normalization (RMS) before final peak limiting.
        """
        rms = float(np.sqrt(np.mean(signal * signal) + 1e-12))
        target_rms = float(10 ** (self.target_rms_dbfs / 20.0))
        if rms > 1e-8:
            signal = signal * (target_rms / rms)
        return signal.astype(np.float32)

    def _peak_target(self, signal: np.ndarray) -> np.ndarray:
        target = 10 ** (self.target_peak_dbfs / 20.0)
        peak = float(np.max(np.abs(signal)))
        if peak > 1e-8:
            signal = signal * (target / peak)
        # Soft limiter to avoid hard clipping artifacts.
        signal = np.tanh(1.15 * signal)
        return np.clip(signal, -1.0, 1.0).astype(np.float32)

    def enhance(self, signal: np.ndarray) -> np.ndarray:
        """
        Run clarity enhancement chain.
        """
        print("[Enhancer] Running DSP speech enhancement...")
        out = signal.astype(np.float32, copy=True)

        if self.enable_bandpass:
            out = self._bandpass_spectral(out)
        if self.enable_gate:
            out = self._spectral_gate(out)
        if self.enable_de_esser:
            out = self._de_esser(out)
        if self.enable_compressor:
            out = self._compress(out)

        out = self._rms_normalize(out)
        out = self._peak_target(out)
        print("[Enhancer] Enhancement complete.")
        return out
