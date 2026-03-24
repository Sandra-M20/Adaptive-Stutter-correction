"""
prolongation_removal.py
=======================
DSP prolongation detection and removal.

Implements correlation-based detection between adjacent frames.
By default it uses report-style thresholding around 14.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np

from prolongation_corrector import ProlongationCorrector


class ProlongationRemover:
    """
    Wrapper over existing DSP prolongation corrector.
    Repetitive frame sequences are implicitly handled via adjacent-frame
    correlation continuity.
    """

    def __init__(
        self,
        sr: int,
        hop_ms: int = 20,
        correlation_threshold: float = 0.75,
        min_prolong_frames: int = 5,
        keep_frames: int = 3,
        max_remove_ratio: float = 0.40,
        use_report_corr14: bool = False,
    ):
        self.corrector = ProlongationCorrector(
            sr=sr,
            hop_ms=hop_ms,
            min_prolong_frames=min_prolong_frames,
            keep_frames=keep_frames,
            max_removal_ratio=max_remove_ratio,
            corr_threshold=correlation_threshold,
            sim_threshold=correlation_threshold,
            use_report_corr14=use_report_corr14,
        )

    def process(self, frames: List[np.ndarray], labels: List[str]) -> Tuple[List[np.ndarray], List[str], Dict]:
        frames, labels, stats = self.corrector.correct(frames, labels)
        print(
            f"[ProlongationRemoval] events={stats.get('prolongation_events', 0)} "
            f"removed_frames={stats.get('frames_removed', 0)}"
        )
        return frames, labels, stats
