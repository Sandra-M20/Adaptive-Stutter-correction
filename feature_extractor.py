"""
feature_extractor.py
================
Minimal FeatureExtractor class to maintain compatibility with existing code.
Provides extract() and extract_batch() methods using the features/ module.
"""

import numpy as np
from typing import List, Dict, Union
from features.mfcc_extractor import MFCCExtractor
from features.lpc_extractor import LPCExtractor
from features.spectral_flux import SpectralFluxExtractor


class FeatureExtractor:
    """
    Minimal FeatureExtractor class for compatibility with prolongation_corrector.py
    """
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.mfcc_extractor = MFCCExtractor(sample_rate=sr)
        self.lpc_extractor = LPCExtractor(sample_rate=sr)
        self.spectral_flux_extractor = SpectralFluxExtractor(sample_rate=sr)
    
    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single frame
        Returns MFCC vector for compatibility with prolongation_corrector.py
        """
        try:
            mfcc_features = self.mfcc_extractor.extract_frame(frame)
            # Return MFCC coefficients as numpy array
            if isinstance(mfcc_features, dict):
                # If it's a dict, extract the MFCC coefficients
                return mfcc_features.get('mfcc', np.zeros(13))
            else:
                # If it's already an array, return it directly
                return mfcc_features
        except Exception:
            return np.zeros(13)  # Default MFCC size
    
    def extract_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract features from multiple frames
        """
        return [self.extract(frame) for frame in frames]
