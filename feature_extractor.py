"""
feature_extractor.py
====================
Pipeline Step 6: Acoustic Feature Extraction (MFCC + LPC)

Converts raw audio frames into compact acoustic feature vectors
used for frame correlation analysis during prolongation detection.

Features extracted:
  1. MFCC (Mel Frequency Cepstral Coefficients)
     - Represents the spectral envelope of speech in a perceptually
       relevant frequency scale.
     - 13 coefficients per frame.

  2. LPC (Linear Predictive Coding)
     - Models the vocal tract transfer function.
     - Captures how the vocal tract shapes the speech sound.
     - 12 coefficients per frame.

Combined feature vector: 25 dimensions per frame (13 + 12).
Higher-dimensional features improve prolongation detection accuracy.
"""

import numpy as np
from config import TARGET_SR, N_MFCC, LPC_ORDER, N_FFT
from utils import compute_mfcc, compute_lpc


class FeatureExtractor:
    """
    Step 6: Extract MFCC + LPC features per speech frame.

    Parameters
    ----------
    sr       : int — Sample rate
    n_mfcc   : int — Number of MFCC coefficients
    lpc_order: int — LPC polynomial order
    n_fft    : int — FFT size for MFCC computation
    """

    def __init__(self,
                 sr: int        = TARGET_SR,
                 n_mfcc: int    = N_MFCC,
                 lpc_order: int = LPC_ORDER,
                 n_fft: int     = N_FFT):
        self.sr        = sr
        self.n_mfcc    = n_mfcc
        self.lpc_order = lpc_order
        self.n_fft     = n_fft
        self.feature_dim = n_mfcc + lpc_order
        # print(f"[FeatureExtractor] Feature vector: {n_mfcc} MFCC + {lpc_order} LPC = {self.feature_dim}D")

    # ------------------------------------------------------------------ #

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract combined MFCC + LPC feature vector for a single frame.

        Parameters
        ----------
        frame : np.ndarray — Single audio frame (1-D)

        Returns
        -------
        features : np.ndarray shape (n_mfcc + lpc_order,)
        """
        mfcc = compute_mfcc(frame, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft)
        lpc  = compute_lpc(frame, order=self.lpc_order)

        mfcc = np.nan_to_num(mfcc)
        lpc  = np.nan_to_num(lpc)

        features = np.concatenate([mfcc, lpc])
        features = features / (np.linalg.norm(features) + 1e-8)
        return features.astype(np.float32)

    def extract_batch(self, frames: list) -> np.ndarray:
        """
        Extract features for a list of frames in batch (much faster).
        """
        if not frames:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        # Vectorized batch extraction
        out = []
        for f in frames:
            # compute_mfcc is now cached and vectorized internally
            mfcc = compute_mfcc(f, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft)
            lpc  = compute_lpc(f, order=self.lpc_order)
            combined = np.concatenate([mfcc, lpc])
            norm = np.linalg.norm(combined) + 1e-8
            out.append(combined / norm)
            
        return np.array(out, dtype=np.float32)

    # ------------------------------------------------------------------ #

    def extract_mfcc_only(self, frame: np.ndarray) -> np.ndarray:
        """Extract only MFCC features (lighter, faster)."""
        return compute_mfcc(frame, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft)

    def extract_lpc_only(self, frame: np.ndarray) -> np.ndarray:
        """Extract only LPC features."""
        return compute_lpc(frame, order=self.lpc_order)

    # ------------------------------------------------------------------ #

    def delta_features(self, features: np.ndarray) -> np.ndarray:
        """
        Compute delta (first-order difference) features for a sequence
        of feature vectors. Delta features capture how features change
        over time — useful for detecting the onset and offset of sounds.

        Parameters
        ----------
        features : np.ndarray shape (T, D)

        Returns
        -------
        delta : np.ndarray shape (T, D)
        """
        if len(features) < 2:
            return np.zeros_like(features)
        delta = np.zeros_like(features)
        delta[1:]  = features[1:] - features[:-1]
        delta[0]   = delta[1]
        return delta

    def delta_delta_features(self, features: np.ndarray) -> np.ndarray:
        """Compute delta-delta (second-order difference) features."""
        return self.delta_features(self.delta_features(features))
