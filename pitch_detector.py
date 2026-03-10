"""
pitch_detector.py
=================
Fundamental Frequency (F0) Pitch Detection

The pitch (F0) of speech is the rate at which the vocal cords vibrate.
It is the most fundamental property of voiced speech.

Why pitch matters for stutter correction?
  - Stuttered blocks often have abnormal pitch contours
  - Pitch discontinuities signal hard onsets and blocks
  - Pitch periodicity validates that a frame is truly voiced speech

Algorithm: Autocorrelation-Based Pitch Detection (YIN-inspired)
  1. Compute autocorrelation of each speech frame: R(tau) = sum(x[n] * x[n+tau])
  2. Compute difference function: d(tau) = R(0) + R(tau) - 2*R(tau)
  3. Compute cumulative mean normalized difference (CMND)
  4. Find the first lag tau where CMND < threshold (default 0.1)
  5. F0 = sr / tau

Mathematical formulation:
  Autocorrelation: R(tau) = sum_{n=0}^{N-1} x[n] * x[n+tau]
  Difference func: d(tau) = sum_{n=0}^{N-1} (x[n] - x[n+tau])^2
  CMND:            d'(tau) = d(tau) / [(1/tau) * sum_{j=1}^{tau} d(j)]
  F0 range:        85 Hz (low male) to 300 Hz (high female child)
"""

import numpy as np
from config import TARGET_SR, FRAME_MS


class PitchDetector:
    """
    F0 (fundamental frequency / pitch) detector using autocorrelation.

    Parameters
    ----------
    sr      : int   — Sample rate
    f_min   : float — Minimum expected F0 (Hz), default 85 Hz
    f_max   : float — Maximum expected F0 (Hz), default 300 Hz
    threshold : float — YIN-like threshold for CMND acceptance
    """

    def __init__(self,
                 sr: int        = TARGET_SR,
                 f_min: float   = 85.0,
                 f_max: float   = 300.0,
                 threshold: float = 0.12):
        self.sr        = sr
        self.f_min     = f_min
        self.f_max     = f_max
        self.threshold = threshold
        # Convert Hz to lag samples
        self.lag_max   = int(sr / f_min) + 1
        self.lag_min   = int(sr / f_max)

    # ------------------------------------------------------------------ #

    def detect_frame(self, frame: np.ndarray) -> float:
        """
        Detect F0 of a single audio frame.

        Parameters
        ----------
        frame : np.ndarray — 1-D audio frame

        Returns
        -------
        f0 : float — Detected fundamental frequency in Hz.
                     Returns 0.0 if frame is unvoiced/silent.
        """
        n    = len(frame)
        lmax = min(self.lag_max, n // 2)

        # Step 1: Difference function
        d = np.zeros(lmax)
        for tau in range(1, lmax):
            diff  = frame[:n - tau] - frame[tau:]
            d[tau] = np.sum(diff ** 2)

        # Step 2: Cumulative mean normalised difference (CMND)
        cmnd    = np.ones(lmax)
        cum_sum = 0.0
        for tau in range(1, lmax):
            cum_sum  += d[tau]
            cmnd[tau] = d[tau] * tau / cum_sum if cum_sum > 0 else 1.0

        # Step 3: Find first minimum below threshold
        tau_est = 0
        for tau in range(self.lag_min, lmax - 1):
            if cmnd[tau] < self.threshold and cmnd[tau] < cmnd[tau + 1]:
                tau_est = tau
                break

        if tau_est == 0:
            return 0.0    # unvoiced frame

        # Step 4: Parabolic interpolation for sub-sample accuracy
        if 0 < tau_est < lmax - 1:
            num   = cmnd[tau_est - 1] - cmnd[tau_est + 1]
            denom = 2 * (cmnd[tau_est - 1] - 2 * cmnd[tau_est] + cmnd[tau_est + 1])
            if abs(denom) > 1e-8:
                tau_est = tau_est + 0.5 * num / denom

        return float(self.sr / tau_est)

    # ------------------------------------------------------------------ #

    def detect_sequence(self, signal: np.ndarray,
                        frame_ms: int = FRAME_MS) -> list:
        """
        Detect F0 for every frame of a signal.

        Parameters
        ----------
        signal   : np.ndarray — full audio signal
        frame_ms : int        — frame size in ms

        Returns
        -------
        f0_contour : list[float] — F0 per frame (0.0 = unvoiced)
        """
        frame_size = int(self.sr * frame_ms / 1000)
        f0s        = []
        for s in range(0, len(signal) - frame_size + 1, frame_size):
            f0 = self.detect_frame(signal[s: s + frame_size])
            f0s.append(f0)
        voiced = sum(1 for f in f0s if f > 0)
        print(f"[PitchDetector] {len(f0s)} frames | Voiced: {voiced} "
              f"({100*voiced/max(len(f0s),1):.1f}%)")
        return f0s

    def voiced_mask(self, f0_contour: list) -> np.ndarray:
        """Boolean mask: True = voiced frame."""
        return np.array([f > 0 for f in f0_contour], dtype=bool)

    def mean_pitch(self, f0_contour: list) -> float:
        """Mean F0 of voiced frames."""
        voiced = [f for f in f0_contour if f > 0]
        return float(np.mean(voiced)) if voiced else 0.0

    def pitch_discontinuities(self, f0_contour: list,
                               jump_hz: float = 50.0) -> list:
        """
        Detect frame indices where pitch jumps by more than `jump_hz` Hz.
        These often correspond to hard onsets or blocks.
        """
        disc = []
        for i in range(1, len(f0_contour)):
            if f0_contour[i] > 0 and f0_contour[i-1] > 0:
                if abs(f0_contour[i] - f0_contour[i-1]) > jump_hz:
                    disc.append(i)
        return disc
