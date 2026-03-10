"""
spectral_flux.py
================
Spectral Flux — Temporal Rate of Spectral Change

Spectral flux measures how rapidly the frequency content of a signal
changes between consecutive analysis frames.

Definition:
  SF(n) = sum_{k} max(0, |X_n(k)|^2 - |X_{n-1}(k)|^2)
  (Positive flux only = onset detection)

Where X_n(k) = k-th bin of STFT frame n.

Why spectral flux matters for stutter correction?
  - High flux → rapid spectral change → onset of a new phoneme
  - Low flux   → no spectral change   → ongoing prolongation
  - Sudden spike in flux after long low-flux region → block release

Applications:
  1. Onset Detection: Detect when new phonemes begin
  2. Repetition Detection: High flux spike followed by identical spectral
     content = word repetition onset
  3. Block Release: Sudden spike after block confirms the blocked sound
     finally erupted
  4. Prolongation Cross-Check: Sustained low flux during speech confirms
     prolongation (phoneme is not changing)

Mathematical background:
  Hann window: w(n) = 0.5 * (1 - cos(2πn/N))
  STFT:        X(n, k) = sum_{m=0}^{N-1} x[m+n*H] * w[m] * e^{-j2πkm/N}
  Power:        P(n, k) = |X(n, k)|^2
  Spectral Flux: SF(n) = sum_{k=0}^{N/2} ReLU(P(n,k) - P(n-1,k))
"""

import numpy as np
from config import TARGET_SR, N_FFT, HOP_STFT
from utils import stft


class SpectralFluxAnalyser:
    """
    Compute and analyse spectral flux for onset and prolongation detection.

    Parameters
    ----------
    sr     : int — Sample rate
    n_fft  : int — FFT size for STFT
    hop    : int — STFT hop size
    """

    def __init__(self,
                 sr: int    = TARGET_SR,
                 n_fft: int = N_FFT,
                 hop: int   = HOP_STFT):
        self.sr    = sr
        self.n_fft = n_fft
        self.hop   = hop

    # ------------------------------------------------------------------ #

    def compute(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute spectral flux over the entire signal.

        Parameters
        ----------
        signal : np.ndarray — mono float32 audio

        Returns
        -------
        flux : np.ndarray shape (T-1,)
               One value per consecutive pair of STFT frames.
        """
        frames = stft(signal, n_fft=self.n_fft, hop=self.hop)
        power  = np.abs(frames) ** 2         # (T, F)

        # Positive spectral flux (onset strength)
        flux = np.sum(np.maximum(power[1:] - power[:-1], 0), axis=1)
        return flux.astype(np.float32)

    # ------------------------------------------------------------------ #

    def detect_onsets(self, flux: np.ndarray,
                      sensitivity: float = 1.5) -> list:
        """
        Detect onset frames where spectral flux exceeds
        mean + sensitivity * std (adaptive threshold).

        Parameters
        ----------
        flux        : np.ndarray — spectral flux array
        sensitivity : float      — multiplier above mean+std

        Returns
        -------
        onset_frames : list[int] — frame indices of detected onsets
        """
        if len(flux) < 3:
            return []
        mean, std = flux.mean(), flux.std()
        threshold = mean + sensitivity * std
        onsets    = [i for i in range(1, len(flux) - 1)
                     if flux[i] > threshold
                     and flux[i] > flux[i - 1]
                     and flux[i] > flux[i + 1]]
        print(f"[SpectralFlux] {len(onsets)} onsets detected "
              f"(thr={threshold:.4f})")
        return onsets

    # ------------------------------------------------------------------ #

    def prolongation_mask(self, flux: np.ndarray,
                           low_thresh_percentile: float = 25.0) -> np.ndarray:
        """
        Boolean mask: True = low flux = ongoing prolongation zone.
        low_thresh_percentile: flux below this percentile = prolonged.
        """
        thr  = np.percentile(flux, low_thresh_percentile)
        mask = flux < thr
        pct  = mask.mean() * 100
        print(f"[SpectralFlux] Prolongation zone: {pct:.1f}% of frames "
              f"(flux < {thr:.4f})")
        return mask

    # ------------------------------------------------------------------ #

    def repetition_candidates(self, flux: np.ndarray,
                               window: int = 5) -> list:
        """
        Identify frame regions where we see high flux (onset)
        followed immediately by low flux (static spectrum).
        This pattern is a strong indicator of word repetition onset.

        Returns
        -------
        candidates : list[int] — starting frame indices of candidate events
        """
        if len(flux) < window * 2:
            return []
        mean  = flux.mean()
        std   = flux.std()
        hi    = mean + std
        lo    = mean - 0.5 * std
        cands = []
        for i in range(len(flux) - window):
            # High flux at i, low flux in next window frames
            if flux[i] > hi and all(flux[i+1:i+window] < lo):
                cands.append(i)
        print(f"[SpectralFlux] Repetition candidates: {len(cands)}")
        return cands

    # ------------------------------------------------------------------ #

    def summary_stats(self, flux: np.ndarray) -> dict:
        """Return summary statistics of the flux array."""
        return {
            "mean":    float(flux.mean()),
            "std":     float(flux.std()),
            "max":     float(flux.max()),
            "min":     float(flux.min()),
            "n_frames": len(flux),
        }
