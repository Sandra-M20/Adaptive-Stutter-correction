"""
segmentation.py
===============
Pipeline Step 3: Speech Segmentation using Short-Time Energy (STE)

Divides the audio signal into fixed-length frames and classifies each
frame as 'speech' or 'silence' based on its Short-Time Energy relative
to an adaptive threshold.

Key concept:
  Short-Time Energy (STE) = mean(frame ^ 2)
  If STE > energy_threshold → speech
  If STE <= energy_threshold → silence
"""

import numpy as np
from config import TARGET_SR, FRAME_MS, HOP_MS, ENERGY_THRESHOLD
from utils import short_time_energy


class SpeechSegmenter:
    """
    Step 3: Segment audio into speech and silence frames using STE.

    Parameters
    ----------
    sr : int
        Sample rate of the audio signal.
    frame_ms : int
        Duration of each analysis frame in milliseconds.
    energy_threshold : float
        STE threshold. Frames with STE > threshold → 'speech'.
    """

    def __init__(self,
                 sr: int = TARGET_SR,
                 frame_ms: int = FRAME_MS,
                 hop_ms: int = HOP_MS,
                 energy_threshold: float = ENERGY_THRESHOLD,
                 auto_threshold: bool = True):
        self.sr               = sr
        self.frame_size       = int(sr * frame_ms / 1000)
        self.hop_size         = int(sr * hop_ms / 1000)
        self.energy_threshold = energy_threshold
        self.auto_threshold   = auto_threshold

    # ------------------------------------------------------------------ #

    def segment(self, signal: np.ndarray):
        """
        Segment `signal` into labeled frames.

        Parameters
        ----------
        signal : np.ndarray
            Mono audio signal at `self.sr`.

        Returns
        -------
        frames   : list[np.ndarray]  — raw sample arrays per frame
        labels   : list[str]         — 'speech' or 'silence' per frame
        energies : list[float]       — STE value per frame
        """
        print(
            f"[Segmentation] Analysing signal ({len(signal)/self.sr:.2f}s) "
            f"| frame={self.frame_size} samples | hop={self.hop_size} samples "
            f"| thr={self.energy_threshold:.4f}"
        )

        frames, labels, energies, zcrs = [], [], [], []

        if len(signal) < self.frame_size:
            energy = short_time_energy(signal)
            return [signal], ["speech"], [energy]

        # Pass 1: Extract all frames and compute energies + ZCR
        for start in range(0, len(signal) - self.frame_size + 1, self.hop_size):
            frame   = signal[start: start + self.frame_size]
            energy  = short_time_energy(frame)
            zcr     = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
            frames.append(frame)
            energies.append(energy)
            zcrs.append(zcr)

        # Use 30th percentile as noise floor — p10 gives falsely tiny thresholds
        arr = np.array(energies, dtype=np.float32)
        valid = arr[arr > 0]
        if self.auto_threshold and len(valid) > 0:
            noise_floor = float(np.percentile(valid, 30))
            self.energy_threshold = max(noise_floor * 1.5, 1e-6)
        elif self.auto_threshold:
            self.energy_threshold = max(float(np.mean(arr)), 1e-6)
        else:
            self.energy_threshold = max(self.energy_threshold, 1e-6)
        if energies:
            if self.auto_threshold:
                print(f"[Segmentation] Auto-calculated threshold (p30): {self.energy_threshold:.6f}")
            else:
                print(f"[Segmentation] Using fixed threshold: {self.energy_threshold:.6f}")

        # Pass 2: Label frames using energy threshold OR ZCR (catches unvoiced consonants)
        for energy, zcr in zip(energies, zcrs):
            if energy > self.energy_threshold or zcr > 0.08:
                label = "speech"
            else:
                label = "silence"
            labels.append(label)

        labels = self._smooth_labels(labels, min_run=2)

        n_speech  = labels.count("speech")
        n_silence = labels.count("silence")
        sp_pct    = n_speech / max(len(labels), 1) * 100

        print(f"[Segmentation] Frames: {len(frames)} "
              f"| Speech: {n_speech} ({sp_pct:.1f}%) "
              f"| Silence: {n_silence}")
        return frames, labels, energies

    def _smooth_labels(self, labels: list, min_run: int = 2) -> list:
        """
        Remove isolated 1-frame label flips to stabilize segmentation.
        """
        if len(labels) < 3:
            return labels
        out = labels[:]
        i = 0
        while i < len(out):
            j = i + 1
            while j < len(out) and out[j] == out[i]:
                j += 1
            run_len = j - i
            if run_len < min_run and i > 0 and j < len(out) and out[i - 1] == out[j]:
                for k in range(i, j):
                    out[k] = out[i - 1]
            i = j
        return out

    # ------------------------------------------------------------------ #

    def get_speech_regions(self, labels: list, energies: list):
        """
        Return start/end frame indices and peak energy for each
        contiguous speech region. Useful for visualization.
        """
        regions = []
        in_speech, start = False, 0
        peak = 0.0
        for i, (lbl, e) in enumerate(zip(labels, energies)):
            if lbl == "speech" and not in_speech:
                in_speech, start, peak = True, i, e
            elif lbl == "speech":
                peak = max(peak, e)
            elif lbl == "silence" and in_speech:
                regions.append({"start": start, "end": i - 1, "peak_energy": peak})
                in_speech = False
        if in_speech:
            regions.append({"start": start, "end": len(labels) - 1, "peak_energy": peak})
        return regions

    # ------------------------------------------------------------------ #

    def adaptive_threshold(self, energies: list,
                           percentile: float = 15.0) -> float:
        """
        Compute an adaptive threshold as the p-th percentile of all
        frame energies. Useful when the default threshold doesn't suit
        a particular recording's volume level.
        """
        arr = np.array(energies, dtype=np.float32)
        valid = arr[arr > 0]
        thr = float(np.percentile(valid, percentile)) if len(valid) > 0 else ENERGY_THRESHOLD
        print(f"[Segmentation] Adaptive threshold (p{percentile:.0f}): {thr:.6f}")
        return thr
