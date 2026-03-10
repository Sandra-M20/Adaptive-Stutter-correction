"""
zero_crossing_rate.py
=====================
Zero-Crossing Rate (ZCR) Analysis for Speech Type Classification

ZCR measures how frequently the audio waveform crosses zero.
It is a classic, computationally lightweight speech feature.

Why ZCR matters?
  - Voiced speech (vowels):   LOW ZCR   (smooth waveform)
  - Unvoiced speech (fricatives like /s/, /f/): HIGH ZCR (noisy waveform)
  - Silence:                  VERY LOW ZCR

Key application to stutter correction:
  - Differentiates prolonged VOICED sounds (e.g. 'aaaa') from
    prolonged UNVOICED sounds (e.g. 'ssss')
  - Combined with STE, gives a 2-D decision space for accurate
    speech/silence/fricative classification

ZCR Formula:
  ZCR(frame) = (1 / (2*N)) * sum_{n=1}^{N} | sign(x[n]) - sign(x[n-1]) |

where N = frame length, x[n] = audio sample.

Combined classification rule (STE + ZCR):
  STE high,  ZCR low  → voiced speech (vowels, nasals)
  STE low,   ZCR high → unvoiced fricatives (/s/, /f/, /sh/)
  STE low,   ZCR low  → silence / pause
  STE high,  ZCR high → stop consonants (/t/, /p/, /k/)
"""

import numpy as np
from config import TARGET_SR, FRAME_MS, ENERGY_THRESHOLD


class ZeroCrossingAnalyser:
    """
    Zero-Crossing Rate analysis for voiced/unvoiced/silence classification.

    Parameters
    ----------
    sr            : int   — Sample rate
    frame_ms      : int   — Frame duration in milliseconds
    zcr_threshold : float — ZCR value above which is classified as unvoiced
    """

    def __init__(self,
                 sr: int            = TARGET_SR,
                 frame_ms: int      = FRAME_MS,
                 zcr_threshold: float = 0.08):
        self.sr            = sr
        self.frame_size    = int(sr * frame_ms / 1000)
        self.zcr_threshold = zcr_threshold

    # ------------------------------------------------------------------ #

    def zcr_frame(self, frame: np.ndarray) -> float:
        """
        Compute Zero-Crossing Rate for a single frame.

        Parameters
        ----------
        frame : np.ndarray — 1-D audio frame

        Returns
        -------
        zcr : float — ZCR in [0.0, 0.5]
        """
        signs = np.sign(frame)
        # Count sign changes (ignore zero-crossings within zero runs)
        crossings = np.sum(np.abs(np.diff(signs))) / 2.0
        return float(crossings / max(len(frame) - 1, 1))

    # ------------------------------------------------------------------ #

    def classify_frame(self, frame: np.ndarray,
                       ste: float = None) -> str:
        """
        Classify a frame using ZCR (and optionally STE).

        Returns
        -------
        label : 'voiced' | 'unvoiced' | 'silence'
        """
        from utils import short_time_energy
        if ste is None:
            ste = short_time_energy(frame)
        zcr = self.zcr_frame(frame)

        if ste < ENERGY_THRESHOLD:
            return "silence"
        elif zcr > self.zcr_threshold:
            return "unvoiced"     # fricatives e.g. /s/ /f/ /sh/
        else:
            return "voiced"       # vowels, nasals

    # ------------------------------------------------------------------ #

    def analyse_signal(self, signal: np.ndarray) -> dict:
        """
        Analyse full signal frame by frame.

        Returns
        -------
        dict with:
          zcr_values   : list[float]
          labels       : list[str]  — 'voiced'/'unvoiced'/'silence'
          stats        : dict       — counts per label
        """
        from utils import short_time_energy
        zcr_values, labels = [], []
        for s in range(0, len(signal) - self.frame_size + 1, self.frame_size):
            frame = signal[s: s + self.frame_size]
            zcr   = self.zcr_frame(frame)
            ste   = short_time_energy(frame)
            label = self.classify_frame(frame, ste)
            zcr_values.append(zcr)
            labels.append(label)

        stats = {k: labels.count(k) for k in ("voiced", "unvoiced", "silence")}
        total = max(len(labels), 1)
        print(f"[ZCR] {len(labels)} frames — "
              f"Voiced: {stats['voiced']} ({100*stats['voiced']/total:.1f}%) | "
              f"Unvoiced: {stats['unvoiced']} ({100*stats['unvoiced']/total:.1f}%) | "
              f"Silence: {stats['silence']} ({100*stats['silence']/total:.1f}%)")
        return {"zcr_values": zcr_values, "labels": labels, "stats": stats}

    # ------------------------------------------------------------------ #

    def is_fricative_prolongation(self, frames: list, labels: list) -> list:
        """
        Identify prolongation blocks that consist of UNVOICED fricatives
        (e.g. repeated /s/ sound as in 'ssssspeech').
        Returns list of (start, end) frame index tuples.
        """
        events = []
        i = 0
        while i < len(frames):
            if labels[i] == "unvoiced":
                start = i
                while i < len(frames) and labels[i] == "unvoiced":
                    i += 1
                if i - start >= 4:    # >= 4 frames (200ms) = prolongation
                    events.append((start, i - 1))
            else:
                i += 1
        return events
