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
                    USE_CONFIDENCE_FILTER, CONFIDENCE_MIN)
from feature_extractor import FeatureExtractor
from utils import cosine_similarity


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
                 sim_threshold: float    = 0.94,
                 min_prolong_frames: int = 7,
                 keep_frames: int        = 4,
                 hop_ms: int             = HOP_MS,
                 max_removal_ratio: float = 0.18,
                 corr_threshold: float = CORR_THRESHOLD,
                 use_report_corr14: bool = USE_REPORT_CORR14,
                 use_confidence_filter: bool = USE_CONFIDENCE_FILTER,
                 min_confidence: float = CONFIDENCE_MIN):
        self.sr                 = sr
        self.sim_threshold      = sim_threshold
        self.min_prolong_frames = min_prolong_frames
        self.keep_frames        = keep_frames
        self.hop_ms             = hop_ms
        self.max_removal_ratio  = max_removal_ratio  # limit: never remove > 30% of a speech run
        self.corr_threshold     = corr_threshold
        self.use_report_corr14  = use_report_corr14
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

    def _adj_spectral_flux(self, run: list) -> np.ndarray:
        """
        Adjacent-frame spectral flux proxy for a speech run.
        """
        if len(run) < 2:
            return np.zeros(len(run), dtype=np.float32)
        flux = np.zeros(len(run), dtype=np.float32)
        for i in range(1, len(run)):
            a = np.abs(np.fft.rfft(run[i - 1])) ** 2
            b = np.abs(np.fft.rfft(run[i])) ** 2
            denom = float(np.mean(a) + 1e-10)
            flux[i] = float(np.mean(np.abs(b - a)) / denom)
        return flux

    # ------------------------------------------------------------------ #

    def correct(self, frames: list, labels: list):
        """
        Apply Steps 5-9 to remove prolongations from all speech runs.

        Parameters
        ----------
        frames : list[np.ndarray]  — frames from PauseCorrector
        labels : list[str]         — 'speech'/'silence' labels

        Returns
        -------
        new_frames  : list[np.ndarray]
        new_labels  : list[str]
        stats       : dict — event counts, frames removed
        """
        print(f"[ProlongationCorrector] sim_threshold={self.sim_threshold:.3f} "
              f"| min_frames={self.min_prolong_frames}"
              f"| corr14={'on' if self.use_report_corr14 else 'off'}")

        new_frames, new_labels = [], []
        total_events    = 0
        total_removed   = 0
        total_rejected  = 0
        i = 0

        while i < len(frames):
            if labels[i] != "speech":
                new_frames.append(frames[i])
                new_labels.append(labels[i])
                i += 1
                continue

            # Collect consecutive speech run
            run = []
            while i < len(frames) and labels[i] == "speech":
                run.append(frames[i])
                i += 1

            if len(run) < 2:
                new_frames.extend(run)
                new_labels.extend(["speech"] * len(run))
                continue
            if len(run) < self.min_prolong_frames:
                new_frames.extend(run)
                new_labels.extend(["speech"] * len(run))
                continue

            # Step 7: feature extraction
            feats = self._extract_features(run)

            # Steps 8-9: detect and remove prolongation blocks
            # Hard cap: never remove > max_removal_ratio of this speech run
            max_frames_to_remove = int(len(run) * self.max_removal_ratio)
            run_removed = 0
            keep_mask = np.ones(len(run), dtype=bool)

            # Precompute adjacent-frame similarities once (vectorized)
            if len(feats) < 2:
                sims = np.zeros(len(run), dtype=np.float32)
            else:
                feats_array = np.array(feats)
                dot = np.sum(feats_array[1:] * feats_array[:-1], axis=1)
                norm1 = np.linalg.norm(feats_array[1:], axis=1)
                norm2 = np.linalg.norm(feats_array[:-1], axis=1)
                sims = np.zeros(len(run), dtype=np.float32)
                sims[1:] = dot / (norm1 * norm2 + 1e-8)
            # Report-style correlation score in [0,20], threshold ~= 14.
            corr_score = (sims + 1.0) * 10.0
            flux = self._adj_spectral_flux(run)
            flux_ref = np.percentile(flux[1:], 45) if len(flux) > 2 else 0.0

            j = 1
            while j < len(run):
                is_candidate = (corr_score[j] >= self.corr_threshold) if self.use_report_corr14 else (sims[j] >= self.sim_threshold)
                if (not is_candidate) or run_removed >= max_frames_to_remove:
                    j += 1
                    continue

                blk_start = j - 1
                while j < len(run):
                    cand_j = (corr_score[j] >= self.corr_threshold) if self.use_report_corr14 else (sims[j] >= self.sim_threshold)
                    if not cand_j:
                        break
                    j += 1
                blk_end = j  # exclusive
                blk_len = blk_end - blk_start

                if blk_len < self.min_prolong_frames:
                    continue

                # Cross-check with spectral stationarity to reduce false removals.
                blk_flux = float(np.mean(flux[max(blk_start + 1, 1):blk_end]))
                if blk_flux > (flux_ref * 1.25 + 1e-8):
                    continue

                # AI-assisted confidence gate: only correct high-confidence disfluencies.
                if self.use_confidence_filter and self._conf_scorer is not None:
                    ev_frames = run[blk_start:blk_end]
                    conf = self._conf_scorer.score(ev_frames)
                    if not conf.get("confident", False):
                        total_rejected += 1
                        continue

                remove_budget = max_frames_to_remove - run_removed
                if remove_budget <= 0:
                    break

                removable = max(0, blk_len - self.keep_frames)
                can_remove = min(removable, remove_budget)
                if can_remove <= 0:
                    continue

                # Keep onset of the phoneme; drop redundant tail frames.
                rm_start = blk_start + self.keep_frames
                rm_end = rm_start + can_remove
                keep_mask[rm_start:rm_end] = False

                total_events += 1
                total_removed += can_remove
                run_removed += can_remove

            kept = [f for idx, f in enumerate(run) if keep_mask[idx]]
            new_frames.extend(kept)
            new_labels.extend(["speech"] * len(kept))

        dur_removed = total_removed * self.hop_ms / 1000
        print(f"[ProlongationCorrector] Events removed: {total_events} "
              f"| Frames removed: {total_removed} (~{dur_removed:.2f}s) "
              f"| Rejected(low-conf): {total_rejected} "
              f"| Max ratio cap: {self.max_removal_ratio:.0%}")

        stats = {
            "prolongation_events": total_events,
            "frames_removed":      total_removed,
            "duration_removed_s":  dur_removed,
            "low_confidence_skipped": total_rejected,
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
