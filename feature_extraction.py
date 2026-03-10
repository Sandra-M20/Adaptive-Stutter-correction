"""
feature_extraction.py
=====================
Frame-level feature extraction for stutter analysis.

Features per frame:
- energy
- amplitude (RMS)
- frequency-spectrum centroid
- MFCC vector
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np

from utils import compute_mfcc


class FrameFeatureExtractor:
    def __init__(self, sr: int, n_mfcc: int = 13):
        self.sr = int(sr)
        self.n_mfcc = int(n_mfcc)

    def extract_one(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        x = frame.astype(np.float32)
        eps = 1e-10
        energy = float(np.mean(x * x))
        amp = float(np.sqrt(max(energy, 0.0)))
        spec = np.abs(np.fft.rfft(x)) + eps
        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.sr)
        centroid = float(np.sum(freqs * spec) / np.sum(spec))
        mfcc = compute_mfcc(x, sr=self.sr, n_mfcc=self.n_mfcc).astype(np.float32)
        return {
            "energy": np.array([energy], dtype=np.float32),
            "amplitude": np.array([amp], dtype=np.float32),
            "spectral_centroid": np.array([centroid], dtype=np.float32),
            "mfcc": mfcc,
        }

    def extract_batch(self, frames: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        return [self.extract_one(f) for f in frames]
