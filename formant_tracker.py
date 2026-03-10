"""
formant_tracker.py
==================
Formant (Resonance) Frequency Tracking

Formants are the resonance frequencies of the vocal tract.
They are the most important features for identifying vowels and
voiced speech sounds.

The three main formants:
  F1 (500-1000 Hz)  — jaw height / vowel openness
  F2 (1000-2500 Hz) — tongue front-back position
  F3 (2500-3500 Hz) — lip rounding / vocal tract shape

Why formants for stutter correction?
  - Prolonged vowels show STATIC F1/F2 (no movement)
  - Normal vowels show DYNAMIC F1/F2 (formant transitions)
  - Blocks show ABRUPT F1/F2 discontinuity
  - By measuring formant stability, we can confirm prolongations

Algorithm:
  1. Extract LPC coefficients for each frame
  2. Compute LPC polynomial roots
  3. Convert roots to frequencies: F = angle(root) * sr / (2*pi)
  4. Keep only roots in voiced speech range (below sr/2)
  5. Sort by frequency → F1, F2, F3

Mathematical basis:
  LPC polynomial: A(z) = 1 - sum_{k=1}^{p} a_k * z^{-k}
  Roots z_k of A(z): F_k = arg(z_k) * fs / (2*pi)
  Bandwidth:        B_k = -ln(|z_k|) * fs / pi
"""

import numpy as np
from config import TARGET_SR, FRAME_MS, LPC_ORDER
from utils import compute_lpc


class FormantTracker:
    """
    Track F1, F2, F3 formant frequencies using LPC root analysis.

    Parameters
    ----------
    sr         : int — Sample rate
    lpc_order  : int — LPC order (should be even, >= 8)
    n_formants : int — Number of formants to extract (default 3)
    """

    def __init__(self,
                 sr: int           = TARGET_SR,
                 lpc_order: int    = max(LPC_ORDER, 12),
                 n_formants: int   = 3):
        self.sr         = sr
        self.lpc_order  = lpc_order
        self.n_formants = n_formants

    # ------------------------------------------------------------------ #

    def extract_formants(self, frame: np.ndarray) -> list:
        """
        Extract formant frequencies from a single audio frame.

        Parameters
        ----------
        frame : np.ndarray — 1-D audio frame (pre-emphasized)

        Returns
        -------
        formants : list[float] — up to n_formants frequencies in Hz.
                                 Padded with 0.0 if fewer are found.
        """
        # Pre-emphasis: emphasizes high frequencies to flatten spectrum
        pre = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])

        lpc_coefs = compute_lpc(pre, order=self.lpc_order)

        # Build the full polynomial [1, -a1, -a2, ..., -ap]
        poly = np.concatenate([[1.0], -lpc_coefs])
        roots = np.roots(poly)

        # Keep only roots in the upper half of the z-plane
        # (complex conjugate pairs → each root represents one formant)
        roots = [r for r in roots if np.imag(r) >= 0]

        # Convert root angle to frequency
        freqs = sorted([
            abs(np.angle(r) * self.sr / (2 * np.pi))
            for r in roots
            if 50 < abs(np.angle(r) * self.sr / (2 * np.pi)) < self.sr / 2
        ])

        # Keep only the lowest n_formants
        result = freqs[:self.n_formants]
        while len(result) < self.n_formants:
            result.append(0.0)
        return result[:self.n_formants]

    # ------------------------------------------------------------------ #

    def track(self, signal: np.ndarray,
              frame_ms: int = FRAME_MS) -> np.ndarray:
        """
        Track formants across the entire signal.

        Parameters
        ----------
        signal   : np.ndarray — full audio signal
        frame_ms : int        — frame duration in ms

        Returns
        -------
        formant_matrix : np.ndarray shape (T_frames, n_formants)
        Each row = [F1, F2, F3] for that frame.
        """
        frame_size = int(self.sr * frame_ms / 1000)
        matrix     = []

        for s in range(0, len(signal) - frame_size + 1, frame_size):
            frame    = signal[s: s + frame_size].astype(np.float64)
            formants = self.extract_formants(frame)
            matrix.append(formants)

        result = np.array(matrix)    # (T, n_formants)
        print(f"[FormantTracker] Tracked {len(result)} frames | "
              f"Mean F1={np.mean(result[:,0]):.0f}Hz "
              f"F2={np.mean(result[:,1]):.0f}Hz "
              f"F3={np.mean(result[:,2]):.0f}Hz")
        return result

    # ------------------------------------------------------------------ #

    def formant_stability(self, formant_matrix: np.ndarray) -> np.ndarray:
        """
        Compute formant stability index per frame as the absolute
        difference of F1 between consecutive frames.

        High stability → prolonged sound (formants not moving)
        Low stability  → normal vowel transition

        Returns
        -------
        stability : np.ndarray shape (T-1,) — F1 change per frame
        """
        if len(formant_matrix) < 2:
            return np.array([])
        return np.abs(np.diff(formant_matrix[:, 0]))    # F1 differences

    def prolongation_by_formant(self, formant_matrix: np.ndarray,
                                 stability_thresh: float = 50.0,
                                 min_frames: int = 4) -> list:
        """
        Detect prolongation events where F1 is static (< stability_thresh Hz/frame).
        Returns list of (start, end) frame tuples.
        """
        stab   = self.formant_stability(formant_matrix)
        stable = stab < stability_thresh
        events, i = [], 0
        while i < len(stable):
            if stable[i]:
                j = i
                while j < len(stable) and stable[j]:
                    j += 1
                if j - i >= min_frames:
                    events.append((i, j))
                i = j
            else:
                i += 1
        return events
