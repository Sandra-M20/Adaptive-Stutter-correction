"""
prolongation_corrector.py
=========================
Pipeline Steps 5, 7, 8, 9: Frame Creation, Correlation Analysis,
Prolongation Detection, and Prolongation Removal.

What is a prolongation?
  A prolongation is when a speaker stretches a sound for longer than
  natural phoneme duration. e.g. "sssssspeech" instead of "speech".
  The stretched frames have nearly identical acoustic features.

Algorithm:
  Step 5 — Divide active speech into overlapping 50ms frames.
  Step 7 — Compute cosine similarity between consecutive frame features.
  Step 8 — If similarity >= threshold for >= min_prolong_frames,
            it is a prolongation event.
  Step 9 — Keep only the first `keep_frames` frames (the real phoneme)
            and discard the redundant stretched frames.
"""

import numpy as np
from config import (TARGET_SR, HOP_MS, SIM_THRESHOLD,
                    MIN_PROLONG_FRAMES, KEEP_FRAMES, PROLONG_MAX_REMOVE_RATIO,
                    CORR_THRESHOLD, USE_REPORT_CORR14,
                    USE_CONFIDENCE_FILTER, CONFIDENCE_MIN,
                    SPECTRAL_FLUX_THRESHOLD, SPECTRAL_FLATNESS_THRESHOLD)
from feature_extractor import FeatureExtractor
from utils import cosine_similarity, spectral_flux, spectral_flatness


class ProlongationCorrector:
    """
    Steps 5, 7, 8, 9: Detect and remove prolonged speech sounds.

    Parameters
    ----------
    sr                  : int   — Sample rate
    sim_threshold       : float — Cosine similarity >= this triggers detection
    min_prolong_frames  : int   — Minimum consecutive similar frames = event
    keep_frames         : int   — Frames to retain from each prolongation block
    """

    def __init__(self,
                 sr: int                 = TARGET_SR,
                 sim_threshold: float    = SIM_THRESHOLD,
                 min_prolong_frames: int = MIN_PROLONG_FRAMES,
                 keep_frames: int        = KEEP_FRAMES,
                 hop_ms: int             = HOP_MS,
                 max_removal_ratio: float = PROLONG_MAX_REMOVE_RATIO,
                 corr_threshold: float = CORR_THRESHOLD,
                 use_report_corr14: bool = USE_REPORT_CORR14,
                 use_confidence_filter: bool = USE_CONFIDENCE_FILTER,
                 min_confidence: float = CONFIDENCE_MIN,
                 flux_threshold: float = SPECTRAL_FLUX_THRESHOLD,
                 flatness_threshold: float = SPECTRAL_FLATNESS_THRESHOLD):
        self.sr                 = sr
        self.sim_threshold      = sim_threshold
        self.min_prolong_frames = min_prolong_frames
        self.keep_frames        = keep_frames
        self.hop_ms             = hop_ms
        self.max_removal_ratio  = max_removal_ratio
        self.corr_threshold     = corr_threshold
        self.use_report_corr14  = use_report_corr14
        self.flux_threshold     = flux_threshold
        self.flatness_threshold = flatness_threshold
        self.extractor          = FeatureExtractor(sr=sr)
        self.use_confidence_filter = use_confidence_filter
        self.min_confidence = min_confidence
        self._conf_scorer = None
        if self.use_confidence_filter:
            try:
                from confidence_scorer import ConfidenceScorer
                self._conf_scorer = ConfidenceScorer(sr=sr, min_confidence=min_confidence)
            except Exception as e:
                print(f"[ProlongationCorrector] Confidence scorer unavailable ({e}); proceeding without it.")
                self.use_confidence_filter = False

    # ------------------------------------------------------------------ #

    def _extract_features(self, run: list) -> list:
        """Step 6/7: Build feature vectors for a speech run (list of frames)."""
        return [self.extractor.extract(f) for f in run]


    # ------------------------------------------------------------------ #

    def correct(self, frames: list, labels: list):
        """
        Apply Steps 5-9 to remove prolongations from all speech runs.
        Uses Multi-Feature detection: Similarity, Flux, Flatness.
        """
        print(f"[ProlongationCorrector] Multi-feature mode: sim={self.sim_threshold:.3f}, "
              f"flux={self.flux_threshold:.4f}, flat={self.flatness_threshold:.3f}")

        new_frames, new_labels = [], []
        total_events    = 0
        total_removed   = 0
        total_rejected  = 0
        detection_events = [] # For evaluation: list of {'start_frame', 'end_frame', 'duration_s'}
        i = 0

        while i < len(frames):
            if labels[i] != "speech":
                new_frames.append(frames[i])
                new_labels.append(labels[i])
                i += 1
                continue

            # Collect consecutive speech run
            run_start_in_full = i
            run = []
            while i < len(frames) and labels[i] == "speech":
                run.append(frames[i])
                i += 1

            if len(run) < self.min_prolong_frames:
                new_frames.extend(run)
                new_labels.extend(["speech"] * len(run))
                continue

            # Feature extraction
            feats = self.extractor.extract_batch(run)
            
            # Multi-feature analysis per frame
            sims = np.zeros(len(run), dtype=np.float32)
            fluxes = np.zeros(len(run), dtype=np.float32)
            flatnesses = np.zeros(len(run), dtype=np.float32)
            
            for j in range(len(run)):
                flatnesses[j] = spectral_flatness(run[j])
                if j > 0:
                    sims[j] = cosine_similarity(feats[j], feats[j-1])
                    fluxes[j] = spectral_flux(run[j-1], run[j])

            # Steps 8-9: detect and remove prolongation blocks
            max_frames_to_remove = int(len(run) * self.max_removal_ratio)
            run_removed = 0
            keep_mask = np.ones(len(run), dtype=bool)

            j = 1
            while j < len(run):
                # All three conditions must be true simultaneously — using AND
                # prevents false positives on normal sustained vowels/sonorants.
                # Natural speech: high similarity AND low flux AND low flatness
                # all occur together only during genuine prolongations.
                is_stutter = (
                    sims[j] >= self.sim_threshold and
                    fluxes[j] <= self.flux_threshold and
                    flatnesses[j] <= self.flatness_threshold
                )

                if (not is_stutter) or run_removed >= max_frames_to_remove:
                    j += 1
                    continue

                blk_start = j - 1
                while j < len(run):
                    is_j_stutter = (
                        sims[j] >= self.sim_threshold and
                        fluxes[j] <= self.flux_threshold and
                        flatnesses[j] <= self.flatness_threshold
                    )
                    if not is_j_stutter:
                        break
                    j += 1
                blk_end = j
                blk_len = blk_end - blk_start

                if blk_len < self.min_prolong_frames:
                    continue


                remove_budget = max_frames_to_remove - run_removed
                if remove_budget <= 0:
                    break

                removable = max(0, blk_len - self.keep_frames)
                can_remove = min(removable, remove_budget)
                if can_remove <= 0:
                    continue

                rm_start = blk_start + self.keep_frames
                rm_end = rm_start + can_remove
                keep_mask[rm_start:rm_end] = False

                total_events += 1
                total_removed += can_remove
                run_removed += can_remove
                
                # Record event for evaluation
                abs_start = run_start_in_full + blk_start
                abs_end = run_start_in_full + blk_end
                detection_events.append({
                    "start_frame": abs_start,
                    "end_frame": abs_end,
                    "duration_s": (blk_end - blk_start) * self.hop_ms / 1000
                })

            kept = [f for idx, f in enumerate(run) if keep_mask[idx]]
            new_frames.extend(kept)
            new_labels.extend(["speech"] * len(kept))

        dur_removed = total_removed * self.hop_ms / 1000
        print(f"[ProlongationCorrector] Events: {total_events} | Removed: {total_removed} (~{dur_removed:.2f}s)")

        stats = {
            "prolongation_events": total_events,
            "frames_removed":      total_removed,
            "duration_removed_s":  dur_removed,
            "low_confidence_skipped": total_rejected,
            "detection_events": detection_events, # Added for evaluation
        }
        return new_frames, new_labels, stats

    # ------------------------------------------------------------------ #

    def similarity_profile(self, frames: list) -> list:
        """
        Return the cosine similarity between every pair of consecutive
        frames. Useful for visualization and threshold calibration.
        """
        if len(frames) < 2:
            return []
        feats = self._extract_features(frames)
        return [cosine_similarity(feats[i], feats[i - 1]) for i in range(1, len(feats))]
