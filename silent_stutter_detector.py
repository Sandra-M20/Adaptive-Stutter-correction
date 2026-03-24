"""
silent_stutter_detector.py
=========================
Minimal SilentStutterDetector class to maintain compatibility with existing code.
Provides basic functionality for silent stutter detection.
"""

import numpy as np
from typing import List, Optional


class SilentStutterDetector:
    """
    Minimal SilentStutterDetector class for compatibility with main_pipeline.py
    """
    
    def __init__(self, sr: int = 16000, hop_ms: int = 12):
        self.sr = sr
        self.hop_ms = hop_ms
        self.hop_samples = int(sr * hop_ms / 1000)
    
    def detect(self, signal: np.ndarray) -> List:
        """
        Detect silent stutters in signal
        Returns empty list for compatibility
        """
        # Minimal implementation - returns no detections
        return []
    
    def detect_batch(self, segments: List) -> List:
        """
        Detect silent stutters in multiple segments
        Returns empty list for compatibility
        """
        return []
    
    def correct(self, frames: List, labels: List) -> tuple:
        """
        Correct silent stutters in frames and labels
        Returns (frames, labels, stats) for compatibility
        """
        stats = {
            'silent_stutters_removed': 0,
            'silent_stutter_frames_removed': 0
        }
        return frames, labels, stats
