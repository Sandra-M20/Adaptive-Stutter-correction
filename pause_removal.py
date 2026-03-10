"""
pause_removal.py
================
DSP long-pause detection and correction.

Rule:
if pause_duration > 0.5s -> compress/remove excessive pause
else keep natural pause
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np

from pause_corrector import PauseCorrector


class LongPauseRemover:
    def __init__(
        self,
        sr: int,
        frame_ms: int = 30,
        hop_ms: int = 15,
        pause_threshold_s: float = 0.5,
        retain_ratio: float = 0.25,
        max_total_removal_ratio: float = 0.10,
    ):
        self.corrector = PauseCorrector(
            sr=sr,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            max_pause_s=pause_threshold_s,
            retain_ratio=retain_ratio,
            max_total_removal_ratio=max_total_removal_ratio,
        )

    def process(self, frames: List[np.ndarray], labels: List[str]) -> Tuple[List[np.ndarray], List[str], Dict]:
        frames, labels, stats = self.corrector.correct(frames, labels)
        print(
            f"[PauseRemoval] pauses={stats.get('pauses_found', 0)} "
            f"removed_frames={stats.get('frames_removed', 0)}"
        )
        return frames, labels, stats
