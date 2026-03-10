"""
confidence_scorer.py
====================
Correction Confidence Scorer

Assigns a confidence score (0-100%) to each correction decision
made by the pipeline. Prevents over-correction of natural speech.

The key problem this solves:
  - Not every prolonged sound is a stutter — some are natural emphasis
  - Not every pause is a block — some are natural breath pauses
  - The system must be conservative: only correct HIGH-CONFIDENCE disfluencies

Confidence Scoring Algorithm:
  For each detected disfluency event, compute a multi-factor score:

  1. Duration Score:      longer = higher confidence it is a disfluency
  2. Similarity Score:   higher cosine similarity = more confident prolongation
  3. Energy Stability:   low variance in energy = more confident prolongation
  4. ZCR Consistency:    consistent ZCR pattern = phoneme is truly repeating
  5. Pitch Stability:    flat F0 = stuck sound, not natural emphasis

  Composite score = weighted sum of factor scores.
  Only events with score > threshold (default 0.6) are corrected.
"""

import numpy as np
from config import TARGET_SR, FRAME_MS
from utils import cosine_similarity, short_time_energy


class ConfidenceScorer:
    """
    Score each disfluency event to prevent over-correction.

    Parameters
    ----------
    sr              : int   — Sample rate
    frame_ms        : int   — Frame size in ms
    min_confidence  : float — Events below this are NOT corrected (0-1)
    weights         : dict  — Factor weights for composite score
    """

    # Default factor weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        "duration":        0.30,
        "similarity":      0.35,
        "energy_stability":0.15,
        "zcr_consistency": 0.10,
        "pitch_stability": 0.10,
    }

    def __init__(self,
                 sr: int               = TARGET_SR,
                 frame_ms: int         = FRAME_MS,
                 min_confidence: float = 0.60,
                 weights: dict         = None):
        self.sr             = sr
        self.frame_ms       = frame_ms
        self.min_confidence = min_confidence
        self.weights        = weights or self.DEFAULT_WEIGHTS

    # ------------------------------------------------------------------ #

    def duration_score(self, n_frames: int) -> float:
        """
        Longer events are more likely disfluencies.
        Score: 0.0 for 1 frame, 1.0 for >= 15 frames (750ms).
        """
        return float(np.clip((n_frames - 1) / 14.0, 0.0, 1.0))

    def similarity_score(self, frames: list) -> float:
        """
        Mean cosine similarity between consecutive frames.
        High similarity = very similar frames = likely prolongation.
        Score normalised from [sim_threshold, 1.0] → [0, 1].
        """
        if len(frames) < 2:
            return 0.0
        from feature_extractor import FeatureExtractor
        fe    = FeatureExtractor(sr=self.sr)
        feats = [fe.extract(f) for f in frames]
        sims  = [cosine_similarity(feats[i], feats[i-1]) for i in range(1, len(feats))]
        mean  = float(np.mean(sims))
        return float(np.clip((mean - 0.50) / 0.50, 0.0, 1.0))

    def energy_stability_score(self, frames: list) -> float:
        """
        Low energy variance within the event = stable (prolonged) sound.
        Score: 1.0 = very stable, 0.0 = very dynamic.
        """
        energies = [short_time_energy(f) for f in frames]
        if not energies:
            return 0.0
        std  = float(np.std(energies))
        mean = float(np.mean(energies)) + 1e-10
        cv   = std / mean       # coefficient of variation
        return float(np.clip(1.0 - cv * 5, 0.0, 1.0))

    def zcr_consistency_score(self, frames: list) -> float:
        """
        Low ZCR variance = consistent phoneme type (all voiced or all unvoiced).
        High ZCR variance = changing phoneme = probably NOT a prolongation.
        """
        from zero_crossing_rate import ZeroCrossingAnalyser
        zca  = ZeroCrossingAnalyser(sr=self.sr)
        zcrs = [zca.zcr_frame(f) for f in frames]
        if not zcrs:
            return 0.0
        std  = float(np.std(zcrs))
        return float(np.clip(1.0 - std * 20, 0.0, 1.0))

    def pitch_stability_score(self, frames: list) -> float:
        """
        Flat pitch contour = stuck on one note = prolongation.
        Score: 1.0 = totally flat, 0.0 = varying pitch.
        """
        from pitch_detector import PitchDetector
        pd  = PitchDetector(sr=self.sr)
        f0s = [pd.detect_frame(f) for f in frames]
        voiced = [f for f in f0s if f > 0]
        if len(voiced) < 2:
            return 0.5       # Unknown — not penalise
        std  = float(np.std(voiced))
        mean = float(np.mean(voiced)) + 1e-10
        cv   = std / mean
        return float(np.clip(1.0 - cv * 3, 0.0, 1.0))

    # ------------------------------------------------------------------ #

    def score(self, frames: list) -> dict:
        """
        Compute a composite confidence score for a candidate event.

        Parameters
        ----------
        frames : list[np.ndarray] — frames of the candidate event

        Returns
        -------
        result : dict — {factor_scores, composite, confident}
        """
        w = self.weights
        factors = {
            "duration":         self.duration_score(len(frames)),
            "similarity":       self.similarity_score(frames),
            "energy_stability": self.energy_stability_score(frames),
            "zcr_consistency":  self.zcr_consistency_score(frames),
            "pitch_stability":  self.pitch_stability_score(frames),
        }
        composite = sum(w[k] * factors[k] for k in factors)
        confident = composite >= self.min_confidence
        return {
            "factors":   factors,
            "composite": round(composite, 4),
            "confident": confident,
        }

    # ------------------------------------------------------------------ #

    def filter_events(self, candidate_events: list,
                      frames: list, labels: list) -> list:
        """
        Filter a list of candidate event dicts (from any detector).
        Each event must have 'start' and 'end' keys (frame indices).

        Returns
        -------
        confirmed_events : list — only events with confidence >= min_confidence
        """
        confirmed = []
        for ev in candidate_events:
            s, e   = ev["start"], ev["end"]
            ev_frs = frames[s:e]
            result = self.score(ev_frs)
            ev["confidence"] = result["composite"]
            if result["confident"]:
                confirmed.append(ev)
            else:
                print(f"[Confidence] Skipped low-confidence event "
                      f"frames {s}-{e} (score={result['composite']:.3f})")
        print(f"[Confidence] {len(confirmed)}/{len(candidate_events)} "
              f"events passed confidence filter.")
        return confirmed
