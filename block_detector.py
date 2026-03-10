"""
block_detector.py
=================
MODULE — Stuttered Block Detection and Correction

A "block" is a sudden complete stoppage of speech mid-word or mid-sentence.
Unlike prolongations (stretched sound) or pauses (silence between words),
blocks appear as:

  1. Sudden energy collapse WITHIN a speech region
  2. Followed by rapid energy recovery as the speaker forces the syllable out

Example: "I w....ant water" (block on 'w')

Detection Algorithm:
  Uses Short-Time Energy (STE) differential analysis within speech regions.

  Step 1: Identify speech frames (non-silence)
  Step 2: Compute frame-by-frame energy delta: ΔE(n) = E(n) - E(n-1)
  Step 3: Detect negative delta spikes → sudden energy drop = block onset
  Step 4: Detect positive delta spikes after the drop → block release
  Step 5: If the energy gap between onset and release > block_threshold,
          classify as a block event

Correction Algorithm:
  Remove the blocked frames (the silent tension period within speech).
  Splice the speech before block directly to speech after block.
  Apply cross-fade to prevent click artifacts.

Mathematical background:
  Energy ratio: R = E_drop / E_pre_block
  If R < block_threshold (default 0.05) → blocked frame
  Block duration: number of consecutive frames below threshold
"""

import numpy as np
from config import (TARGET_SR, FRAME_MS, ENERGY_THRESHOLD,
                    BLOCK_MAX_REMOVE_RATIO, BLOCK_KEEP_RATIO,
                    BLOCK_MAX_FRAMES, BLOCK_CONTEXT_FRAMES,
                    BLOCK_RECOVERY_RATIO)
from utils import short_time_energy


class BlockDetector:
    """
    Detect and remove stuttered blocks (hard stops within speech).

    Parameters
    ----------
    sr              : int   — Sample rate
    frame_ms        : int   — Frame length in ms
    block_threshold : float — Energy ratio threshold (frames below this = blocked)
    min_block_frames: int   — Minimum frames below threshold to count as a block
    """

    def __init__(self,
                 sr: int                = TARGET_SR,
                 frame_ms: int          = FRAME_MS,
                 block_threshold: float = 0.05,
                 min_block_frames: int  = 3,
                 max_remove_ratio: float = BLOCK_MAX_REMOVE_RATIO,
                 keep_ratio: float = BLOCK_KEEP_RATIO,
                 max_block_frames: int = BLOCK_MAX_FRAMES,
                 context_frames: int = BLOCK_CONTEXT_FRAMES,
                 recovery_ratio: float = BLOCK_RECOVERY_RATIO):
        self.sr              = sr
        self.frame_size      = int(sr * frame_ms / 1000)
        self.frame_ms        = frame_ms
        self.block_threshold = block_threshold
        self.min_block_frames= min_block_frames
        self.max_remove_ratio = max_remove_ratio
        self.keep_ratio       = keep_ratio
        self.max_block_frames = max_block_frames
        self.context_frames   = context_frames
        self.recovery_ratio   = recovery_ratio

    # ------------------------------------------------------------------ #

    def _energy_profile(self, frames: list) -> np.ndarray:
        """Compute normalized STE per frame."""
        energies = np.array([short_time_energy(f) for f in frames])
        if energies.max() > 1e-10:
            energies /= energies.max()
        return energies

    # ------------------------------------------------------------------ #

    def detect(self, frames: list, labels: list) -> list:
        """
        Detect block events in the frame sequence.

        Parameters
        ----------
        frames : list[np.ndarray] — audio frames
        labels : list[str]        — 'speech'/'silence' labels

        Returns
        -------
        blocks : list[dict] — each dict has 'start', 'end', 'duration_s'
        """
        energies = self._energy_profile(frames)
        blocks   = []
        i        = 0

        while i < len(frames):
            if labels[i] != "speech":
                i += 1
                continue

            # Find a run of speech
            run_start = i
            while i < len(frames) and labels[i] == "speech":
                i += 1
            run_end = i

            # Within this speech run, find energy drops
            run_e = energies[run_start:run_end]
            if len(run_e) < 4:
                continue

            # Normalize within the speech run
            run_max = run_e.max()
            if run_max < 1e-10:
                continue
            norm_e = run_e / run_max

            # Detect blocks: frames well below mean energy within speech run
            mean_e = norm_e.mean()
            j      = 0
            while j < len(norm_e):
                if norm_e[j] < self.block_threshold:
                    blk_start = j
                    while j < len(norm_e) and norm_e[j] < self.block_threshold:
                        j += 1
                    blk_len = j - blk_start
                    if blk_len >= self.min_block_frames and blk_len <= self.max_block_frames:
                        left = max(0, blk_start - self.context_frames)
                        right = min(len(norm_e), j + self.context_frames)
                        pre = norm_e[left:blk_start]
                        post = norm_e[j:right]
                        if len(pre) == 0 or len(post) == 0:
                            continue
                        pre_mean = float(np.mean(pre))
                        post_mean = float(np.mean(post))
                        # Require a clear "stuck then release" pattern.
                        if pre_mean < (self.block_threshold * 1.2):
                            continue
                        if post_mean < (pre_mean * self.recovery_ratio):
                            continue
                        abs_start = run_start + blk_start
                        abs_end   = run_start + j
                        dur       = blk_len * self.frame_ms / 1000
                        blocks.append({
                            "start":      abs_start,
                            "end":        abs_end,
                            "duration_s": dur,
                        })
                else:
                    j += 1

        if blocks:
            print(f"[BlockDetector] Detected {len(blocks)} block(s). "
                  f"Total: {sum(b['duration_s'] for b in blocks):.2f}s")
        else:
            print("[BlockDetector] No blocks detected.")
        return blocks

    # ------------------------------------------------------------------ #

    def correct(self, frames: list, labels: list):
        """
        Remove detected blocks from the frame sequence.

        Parameters
        ----------
        frames : list[np.ndarray]
        labels : list[str]

        Returns
        -------
        new_frames : list[np.ndarray]
        new_labels : list[str]
        stats      : dict
        """
        blocks = self.detect(frames, labels)
        if not blocks:
            return frames, labels, {"blocks_removed": 0}

        # Compress (not fully delete) detected blocks with a global cap.
        remove = set()
        max_total_remove = int(len(frames) * self.max_remove_ratio)
        removed_so_far = 0
        for b in blocks:
            if removed_so_far >= max_total_remove:
                break
            blk_len = max(0, b["end"] - b["start"])
            keep_n = int(np.ceil(blk_len * self.keep_ratio))
            rm_start = b["start"] + min(keep_n, blk_len)
            rm_end = b["end"]
            for idx in range(rm_start, rm_end):
                if removed_so_far >= max_total_remove:
                    break
                remove.add(idx)
                removed_so_far += 1

        new_frames = [f for i, f in enumerate(frames) if i not in remove]
        new_labels = [l for i, l in enumerate(labels) if i not in remove]

        print(f"[BlockDetector] Removed {len(remove)} blocked frames "
              f"({len(remove)*self.frame_ms/1000:.2f}s).")
        return new_frames, new_labels, {
            "blocks_removed": len(blocks),
            "frames_removed": len(remove),
            "duration_removed_s": len(remove) * self.frame_ms / 1000.0,
        }

    # ------------------------------------------------------------------ #

    def energy_delta_profile(self, frames: list) -> np.ndarray:
        """
        Return the frame-by-frame energy delta (ΔE).
        Useful for visualization and threshold calibration.
        """
        energies = self._energy_profile(frames)
        delta    = np.zeros_like(energies)
        delta[1:] = np.diff(energies)
        return delta
