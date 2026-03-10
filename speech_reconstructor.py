"""
speech_reconstructor.py
=======================
Pipeline Step 11: Corrected Speech Reconstruction via Overlap-Add Synthesis

After prolongation and pause correction, the speech is in the form of
modified frame sequences. This module stitches them back into a
continuous audio signal without audible clicks or discontinuities.

Technique: Overlap-Add (OLA)
  - Each frame is windowed with a Hann window
  - Frames are placed with 50% overlap
  - Overlapping regions are summed, then re-normalized by the window sum
  - This ensures smooth transitions between adjacent frames

Why OLA?
  Simple frame concatenation creates hard boundaries that produce
  click artifacts at frame edges. OLA uses overlapping windows to
  create smooth cross-fades between adjacent frames.
"""

import numpy as np
from config import TARGET_SR, FRAME_MS, HOP_MS
from utils import normalize


class SpeechReconstructor:
    """
    Step 11: Reconstruct a continuous waveform from corrected frames.

    Parameters
    ----------
    sr          : int   — Sample rate
    overlap     : float — Fractional overlap for OLA (0.5 = 50%)
    """

    def __init__(self,
                 sr: int          = TARGET_SR,
                 frame_ms: int    = FRAME_MS,
                 hop_ms: int      = HOP_MS):
        self.sr         = sr
        self.frame_size = int(sr * frame_ms / 1000)
        self.hop        = int(sr * hop_ms / 1000)
        self.overlap    = 1.0 - (self.hop / max(self.frame_size, 1))

    # ------------------------------------------------------------------ #

    def reconstruct(self, frames: list, labels: list) -> np.ndarray:
        """
        Overlap-Add reconstruction of speech frames.

        Parameters
        ----------
        frames : list[np.ndarray] — corrected frames
        labels : list[str]        — 'speech' / 'silence' labels (unused but kept for API)

        Returns
        -------
        signal : np.ndarray — reconstructed mono audio signal
        """
        if not frames:
            return np.zeros(1, dtype=np.float32)

        frame_size = len(frames[0])
        hop        = self.hop

        # Output length
        out_len    = hop * (len(frames) - 1) + frame_size

        output     = np.zeros(out_len, dtype=np.float64)
        window_sum = np.zeros(out_len, dtype=np.float64)
        window     = np.hamming(frame_size)

        for i, frame in enumerate(frames):
            start = i * hop
            end   = start + frame_size
            # Pad/trim to expected frame size
            if len(frame) < frame_size:
                frame = np.concatenate([frame, np.zeros(frame_size - len(frame), dtype=np.float32)])
            elif len(frame) > frame_size:
                frame = frame[:frame_size]
            win_frame          = frame * window
            output[start:end]     += win_frame
            window_sum[start:end] += window

        # Normalize overlap
        nonzero          = window_sum > 1e-6
        output[nonzero] /= window_sum[nonzero]

        result = output.astype(np.float32)
        dur = len(result) / self.sr
        print(f"[Reconstruction] OLA complete. Duration: {dur:.3f}s | Frames: {len(frames)}")
        return result

    def _smooth_splice_edges(self,
                             signal: np.ndarray,
                             win_samples: int = 12,
                             max_smooth_points: int = 60) -> np.ndarray:
        """
        Reduce audible roughness at cut/splice points by smoothing very large
        sample-to-sample jumps. This acts as a lightweight de-clicker.
        """
        x = signal.astype(np.float32, copy=True)
        if len(x) < (2 * win_samples + 3):
            return x

        d = np.abs(np.diff(x))
        med = float(np.median(d)) + 1e-8
        thr = max(0.45, 12.0 * med)
        jump_idx = np.where(d > thr)[0]
        if len(jump_idx) == 0:
            return x

        applied = 0
        last_idx = -10 * win_samples
        for j in jump_idx:
            if applied >= max_smooth_points:
                break
            if j - last_idx < win_samples:
                continue
            left = j - win_samples
            right = j + win_samples + 1
            if left < 0 or right >= len(x):
                continue
            ramp = np.linspace(x[left], x[right], right - left + 1, dtype=np.float32)
            x[left:right + 1] = 0.7 * ramp + 0.3 * x[left:right + 1]
            applied += 1
            last_idx = j

        if applied > 0:
            print(f"[Reconstruction] Smoothed {applied} splice edge(s).")
        return x

    # ------------------------------------------------------------------ #

    def simple_concat(self, frames: list) -> np.ndarray:
        """
        Naive frame concatenation (no overlap). Produces click artifacts
        at frame boundaries. Provided for comparison / debugging.
        """
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).astype(np.float32)

    def crossfade(self, frames: list, fade_ms: int = 5) -> np.ndarray:
        """
        Concatenate frames with a short linear crossfade at each
        boundary to reduce but not fully eliminate click artifacts.
        Lighter alternative to full OLA.
        """
        if not frames:
            return np.zeros(0, dtype=np.float32)
        fade_n = int(self.sr * fade_ms / 1000)
        result = frames[0].copy().astype(np.float32)
        for frame in frames[1:]:
            frame = frame.astype(np.float32)
            if len(result) < fade_n or len(frame) < fade_n:
                result = np.concatenate([result, frame])
                continue
            ramp_out = np.linspace(1, 0, fade_n, dtype=np.float32)
            ramp_in  = np.linspace(0, 1, fade_n, dtype=np.float32)
            result[-fade_n:] = result[-fade_n:] * ramp_out + frame[:fade_n] * ramp_in
            result = np.concatenate([result, frame[fade_n:]])
        return normalize(result)
