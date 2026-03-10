"""
dataset_loader.py
=================
UCLASS (University College London Archive of Stuttered Speech) Dataset Loader

Reads the archive's `labels.csv` and loads clips with their multi-label
binary annotations for use in training and evaluation.

Dataset structure:
  archive/clips/labels.csv  — CSV with one row per clip
  archive/clips/clips/      — 3-second WAV files at 16kHz

Label columns:
  Block, Prolongation, SoundRep, WordRep, Interjection, NoStutteredWords

This module provides:
  - DatasetLoader.load()              — returns a list of sample dicts
  - DatasetLoader.get_by_type()       — filter by dysfluency type
  - DatasetLoader.stats()             — print dataset statistics
  - DatasetLoader.sample_batch()      — randomly sample a mini-batch
"""

import os
import csv
import json
import random
import numpy as np
import soundfile as sf
from config import CLIPS_CSV, CLIPS_DIR, ARCHIVE_DIR, LABEL_COLS


class DatasetLoader:
    """
    Load and serve audio clips from the UCLASS archive.

    Parameters
    ----------
    csv_path   : str — path to labels.csv
    clips_root : str — root directory for resolving relative clip paths
    sr         : int — target sample rate for loaded audio
    """

    def __init__(self,
                 csv_path: str   = CLIPS_CSV,
                 clips_root: str = "archive",
                 sr: int         = 16000):
        self.csv_path   = csv_path
        self.clips_root = clips_root
        self.sr         = sr
        self._samples   = None   # Lazy-loaded

    # ------------------------------------------------------------------ #

    def load(self, max_samples: int = None) -> list:
        """
        Parse labels.csv and build a list of sample dictionaries.

        Each sample dict contains:
          {
            "path":            str  — absolute path to WAV file,
            "Block":           int  — 1/0,
            "Prolongation":    int  — 1/0,
            "SoundRep":        int  — 1/0,
            "WordRep":         int  — 1/0,
            "Interjection":    int  — 1/0,
            "NoStutteredWords":int  — 1/0,
            "is_dysfluent":    bool — True if any dysfluency label is 1,
          }

        Parameters
        ----------
        max_samples : int or None — max samples to load (None = all)
        """
        if not os.path.exists(self.csv_path):
            print(f"[Dataset] ERROR: labels.csv not found at '{self.csv_path}'")
            return []

        samples = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["filepath"]
                abs_path = os.path.join(self.clips_root, rel_path)
                if not os.path.exists(abs_path):
                    continue
                labels  = {col: int(row[col]) for col in LABEL_COLS if col in row}
                is_dys  = any(v == 1 for k, v in labels.items() if k != "NoStutteredWords")
                sample  = {"path": abs_path, **labels, "is_dysfluent": is_dys}
                samples.append(sample)
                if max_samples and len(samples) >= max_samples:
                    break

        self._samples = samples
        print(f"[Dataset] Loaded {len(samples)} clips from labels.csv.")
        return samples

    # ------------------------------------------------------------------ #

    def get_samples(self) -> list:
        """Return cached samples, loading if needed."""
        if self._samples is None:
            self.load()
        return self._samples

    def get_by_type(self, dysfluency_type: str) -> list:
        """
        Filter samples by a specific dysfluency label.

        Parameters
        ----------
        dysfluency_type : one of 'Block', 'Prolongation', 'SoundRep',
                          'WordRep', 'Interjection', 'NoStutteredWords'
        """
        if dysfluency_type not in LABEL_COLS:
            raise ValueError(f"Invalid type. Must be one of {LABEL_COLS}")
        return [s for s in self.get_samples() if s.get(dysfluency_type) == 1]

    # ------------------------------------------------------------------ #

    def sample_batch(self, n: int, dysfluent_only: bool = False,
                     seed: int = None) -> list:
        """
        Randomly sample `n` clips.

        Parameters
        ----------
        n              : int  — number of clips to return
        dysfluent_only : bool — if True, only return stuttered clips
        seed           : int  — random seed for reproducibility
        """
        pool = self.get_samples()
        if dysfluent_only:
            pool = [s for s in pool if s["is_dysfluent"]]
        if seed is not None:
            random.seed(seed)
        return random.sample(pool, min(n, len(pool)))

    # ------------------------------------------------------------------ #

    def load_audio(self, sample: dict) -> tuple:
        """
        Load the audio waveform for a sample dict.

        Returns
        -------
        signal : np.ndarray — mono float32 audio
        sr     : int        — sample rate
        """
        signal, sr = sf.read(sample["path"], dtype="float32", always_2d=False)
        if signal.ndim == 2:
            signal = signal.mean(axis=1)
        if sr != self.sr:
            from utils import resample
            signal = resample(signal, sr, self.sr)
        return signal, self.sr

    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """Print and return dataset statistics."""
        samples = self.get_samples()
        total   = len(samples)
        counts  = {col: sum(1 for s in samples if s.get(col) == 1) for col in LABEL_COLS}
        dysfluent = sum(1 for s in samples if s["is_dysfluent"])
        fluent    = total - dysfluent

        print("\n=== UCLASS Dataset Statistics ===")
        print(f"  Total clips   : {total}")
        print(f"  Dysfluent     : {dysfluent} ({100*dysfluent/max(total,1):.1f}%)")
        print(f"  Fluent        : {fluent}  ({100*fluent/max(total,1):.1f}%)")
        print("  Per-label counts:")
        for col, n in counts.items():
            print(f"    {col:20s}: {n:4d}  ({100*n/max(total,1):.1f}%)")
        print("=================================\n")

        return {"total": total, "dysfluent": dysfluent, "fluent": fluent, "labels": counts}
