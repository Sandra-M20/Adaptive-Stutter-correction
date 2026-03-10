"""
pause_corrector.py
==================
Pipeline Step 4: Long Pause Detection and Removal

In fluent speech, short pauses (~100-200ms) naturally occur between
words and phrases. PWS (Persons Who Stutter) often produce abnormally
long silent pauses that disrupt the flow of speech.

This module:
  1. Scans the frame-level labels from the Segmenter.
  2. Identifies runs of consecutive silence frames that exceed a
     maximum allowed pause duration.
  3. Compresses each detected long pause, retaining a small fraction
     to preserve natural speech rhythm.

Algorithm parameters are loaded from config.py and can be overridden
by the Reptile MAML adaptive optimizer (Step 10).
"""

import math
import numpy as np
from config import (TARGET_SR, FRAME_MS, HOP_MS, MAX_PAUSE_S,
                    PAUSE_RETAIN_RATIO, PAUSE_MAX_REMOVE_RATIO)


class PauseCorrector:
    """
    Step 4: Detect and compress abnormally long silence gaps.

    Parameters
    ----------
    sr              : int   — Sample rate
    frame_ms        : int   — Frame length in milliseconds
    max_pause_s     : float — Silence runs longer than this are considered
                              stuttered pauses (in seconds).
    retain_ratio    : float — Fraction of the pause to keep (0.10 = 10%).
    """

    def __init__(self,
                 sr: int           = TARGET_SR,
                 frame_ms: int     = FRAME_MS,
                 hop_ms: int       = HOP_MS,
                 max_pause_s: float = MAX_PAUSE_S,
                 retain_ratio: float = PAUSE_RETAIN_RATIO,
                 max_total_removal_ratio: float = PAUSE_MAX_REMOVE_RATIO):
        self.sr               = sr
        self.frame_ms         = frame_ms
        self.hop_ms           = hop_ms
        self.max_pause_frames = math.ceil(max_pause_s / (hop_ms / 1000))
        self.retain_ratio     = retain_ratio
        self.max_pause_s      = max_pause_s
        self.max_total_removal_ratio = max_total_removal_ratio

    # ------------------------------------------------------------------ #

    def correct(self, frames: list, labels: list):
        """
        Compress abnormally long silence runs in the frame sequence.

        Parameters
        ----------
        frames : list[np.ndarray]   — Audio frames from Segmenter
        labels : list[str]          — Corresponding 'speech'/'silence' labels

        Returns
        -------
        new_frames : list[np.ndarray]
        new_labels : list[str]
        stats      : dict — {'pauses_found': int, 'frames_removed': int}
        """
        print(f"[PauseCorrector] Scanning for pauses > {self.max_pause_s}s "
              f"(>{self.max_pause_frames} frames)...")

        new_frames, new_labels = [], []
        pauses_found = 0
        frames_removed = 0
        max_total_remove = int(len(frames) * self.max_total_removal_ratio)
        i = 0

        while i < len(frames):
            if labels[i] == "silence":
                # Collect entire silence run
                run_start = i
                while i < len(labels) and labels[i] == "silence":
                    i += 1
                run_len = i - run_start

                # Ignore tiny pauses; they are natural transitions.
                if run_len < 3:
                    new_frames.extend(frames[run_start:i])
                    new_labels.extend(labels[run_start:i])
                    continue

                is_internal_pause = run_start > 0 and i < len(labels)
                remaining_budget = max_total_remove - frames_removed

                if run_len > self.max_pause_frames and is_internal_pause and remaining_budget > 0:
                    # Long pause detected — compress
                    keep_n = max(1, int(run_len * self.retain_ratio))
                    keep_n = min(keep_n, run_len)
                    keep_n = max(keep_n, run_len - remaining_budget)
                    new_frames.extend(frames[run_start: run_start + keep_n])
                    new_labels.extend(["silence"] * keep_n)
                    pauses_found  += 1
                    frames_removed += (run_len - keep_n)
                else:
                    # Normal short pause — keep as-is
                    new_frames.extend(frames[run_start:i])
                    new_labels.extend(labels[run_start:i])
            else:
                new_frames.append(frames[i])
                new_labels.append(labels[i])
                i += 1

        pause_dur_removed = frames_removed * (self.hop_ms / 1000)
        print(
            f"[PauseCorrector] pauses={pauses_found} "
            f"removed_frames={frames_removed} "
            f"removed_duration={pause_dur_removed:.2f}s"
        )

        stats = {
            "pauses_found":    pauses_found,
            "frames_removed":  frames_removed,
            "duration_removed_s": pause_dur_removed,
        }
        return new_frames, new_labels, stats

    # ------------------------------------------------------------------ #

    def detect_only(self, labels: list):
        """
        Return a list of (start_frame, end_frame, duration_s) tuples
        for all long pauses without modifying the signal.
        Useful for analysis / visualization.
        """
        pauses = []
        i = 0
        while i < len(labels):
            if labels[i] == "silence":
                run_start = i
                while i < len(labels) and labels[i] == "silence":
                    i += 1
                run_len = i - run_start
                if run_len > self.max_pause_frames:
                    dur = run_len * self.frame_ms / 1000
                    pauses.append({"start": run_start, "end": i - 1, "duration_s": dur})
            else:
                i += 1
        return pauses
