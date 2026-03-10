"""
silent_stutter_detector.py
==========================
AI-assisted silent stutter detector for whole-audio analysis.

Targets internal silent dysfluencies such as brief speech blocks/hesitations:
  - "I ... want" where silence appears inside an utterance and is too short
    or irregular to be captured by standard long-pause compression alone.

Approach:
  1. Scan contiguous silence runs in frame labels.
  2. Keep only internal runs (speech -> silence -> speech).
  3. Compute confidence from duration + valley depth + context continuity.
  4. Compress high-confidence runs using a retention ratio.
"""

from __future__ import annotations

import math
import numpy as np

from config import (
    TARGET_SR,
    HOP_MS,
    SILENT_STUTTER_MIN_S,
    SILENT_STUTTER_MAX_S,
    SILENT_STUTTER_KEEP,
    SILENT_STUTTER_CONF,
    SILENT_STUTTER_DSP_MIN,
    SILENT_STUTTER_MAX_REMOVE_RATIO,
)
from utils import short_time_energy


class SilentStutterDetector:
    """
    Detect and compress silent stutter events across full audio.
    """

    def __init__(
        self,
        sr: int = TARGET_SR,
        hop_ms: int = HOP_MS,
        min_s: float = SILENT_STUTTER_MIN_S,
        max_s: float = SILENT_STUTTER_MAX_S,
        keep_ratio: float = SILENT_STUTTER_KEEP,
        min_confidence: float = SILENT_STUTTER_CONF,
        min_dsp_score: float = SILENT_STUTTER_DSP_MIN,
        max_total_removal_ratio: float = SILENT_STUTTER_MAX_REMOVE_RATIO,
    ):
        self.sr = sr
        self.hop_ms = hop_ms
        self.min_frames = max(1, int(round(min_s / (hop_ms / 1000.0))))
        self.max_frames = max(self.min_frames, int(round(max_s / (hop_ms / 1000.0))))
        self.keep_ratio = float(np.clip(keep_ratio, 0.1, 0.95))
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.min_dsp_score = float(np.clip(min_dsp_score, 0.0, 1.0))
        self.max_total_removal_ratio = max_total_removal_ratio
        self._relaxed_min_frames = max(1, self.min_frames - 2)

    def _mean_energy(self, seq: list[np.ndarray]) -> float:
        if not seq:
            return 0.0
        return float(np.mean([short_time_energy(f) for f in seq]))

    def _event_features(
        self,
        frames: list[np.ndarray],
        run_start: int,
        run_end: int,
    ) -> dict:
        run_len = run_end - run_start
        dur_score = float(np.clip((run_len - self.min_frames) / max(self.max_frames - self.min_frames, 1), 0.0, 1.0))

        left = frames[max(0, run_start - 3):run_start]
        mid = frames[run_start:run_end]
        right = frames[run_end:min(len(frames), run_end + 3)]

        e_left = self._mean_energy(left)
        e_mid = self._mean_energy(mid)
        e_right = self._mean_energy(right)
        ctx = 0.5 * (e_left + e_right) + 1e-10
        valley_ratio = e_mid / ctx
        depth_score = float(np.clip((0.40 - valley_ratio) / 0.40, 0.0, 1.0))

        continuity = 1.0 if (e_left > 1e-5 and e_right > 1e-5) else 0.0

        return {
            "dur_score": dur_score,
            "depth_score": depth_score,
            "continuity": continuity,
        }

    def _dsp_score(self, features: dict) -> float:
        # DSP-rules score: duration+depth are dominant; continuity helps.
        score = 0.50 * features["dur_score"] + 0.40 * features["depth_score"] + 0.10 * features["continuity"]
        return float(np.clip(score, 0.0, 1.0))

    def _ai_score(self, features: dict) -> float:
        # AI-style confidence from the same robust feature set with different weighting.
        score = 0.35 * features["dur_score"] + 0.45 * features["depth_score"] + 0.20 * features["continuity"]
        return float(np.clip(score, 0.0, 1.0))

    def detect(self, frames: list[np.ndarray], labels: list[str]) -> list[dict]:
        events = []
        i = 0
        while i < len(labels):
            if labels[i] != "silence":
                i += 1
                continue

            s = i
            while i < len(labels) and labels[i] == "silence":
                i += 1
            e = i
            run_len = e - s

            is_internal = s > 0 and e < len(labels) and labels[s - 1] == "speech" and labels[e] == "speech"
            if not is_internal:
                continue
            if run_len > self.max_frames:
                continue

            features = self._event_features(frames, s, e)
            dsp_score = self._dsp_score(features)
            ai_score = self._ai_score(features)
            # Relax duration requirement for very clear micro-silent blocks.
            short_micro_block = (
                run_len >= self._relaxed_min_frames
                and run_len < self.min_frames
                and features["depth_score"] >= 0.75
                and features["continuity"] >= 0.90
            )
            if not short_micro_block and run_len < self.min_frames:
                continue

            if short_micro_block:
                dsp_score = min(1.0, dsp_score + 0.12)
                ai_score = min(1.0, ai_score + 0.12)

            if dsp_score < self.min_dsp_score or ai_score < self.min_confidence:
                continue

            events.append(
                {
                    "start": s,
                    "end": e,
                    "frames": run_len,
                    "duration_s": run_len * (self.hop_ms / 1000.0),
                    "dsp_score": round(dsp_score, 4),
                    "ai_score": round(ai_score, 4),
                }
            )

        if events:
            print(
                f"[SilentStutterAI] Detected {len(events)} dual-confirmed event(s) "
                f"(AI>={self.min_confidence:.2f}, DSP>={self.min_dsp_score:.2f})."
            )
        else:
            print("[SilentStutterAI] No dual-confirmed silent stutter events detected.")
        return events

    def correct(self, frames: list[np.ndarray], labels: list[str]):
        events = self.detect(frames, labels)
        if not events:
            return frames, labels, {"silent_stutters_removed": 0, "frames_removed": 0}

        remove = set()
        max_total_remove = int(len(frames) * self.max_total_removal_ratio)
        removed = 0
        for ev in events:
            s, e = ev["start"], ev["end"]
            run_len = e - s
            keep_n = max(1, int(math.ceil(run_len * self.keep_ratio)))
            rem_start = s + keep_n
            for idx in range(rem_start, e):
                if removed >= max_total_remove:
                    break
                if idx not in remove:
                    remove.add(idx)
                    removed += 1
            if removed >= max_total_remove:
                break

        new_frames = [f for i, f in enumerate(frames) if i not in remove]
        new_labels = [l for i, l in enumerate(labels) if i not in remove]
        print(
            f"[SilentStutterAI] Removed {removed} frame(s) "
            f"(~{removed * self.hop_ms / 1000.0:.2f}s, cap {self.max_total_removal_ratio:.0%})."
        )
        return new_frames, new_labels, {
            "silent_stutters_removed": len(events),
            "frames_removed": removed,
            "duration_removed_s": removed * self.hop_ms / 1000.0,
            "dual_confirmed_events": len(events),
        }
