"""
adaptive_optimizer.py
=====================
Pipeline Step 10: Adaptive Threshold Optimization via Reptile MAML

This module implements the Reptile meta-learning algorithm
(Nichol et al. 2018) applied to the DSP threshold parameter space.

Why adaptive thresholds?
  Different speakers have different voices, speaking speeds, and
  stuttering patterns. A single fixed threshold set cannot work
  optimally for all speakers.

How Reptile MAML works here:
  The "parameters" being optimized are:
    - energy_threshold  (STE segmentation)
    - max_pause_s       (pause correction)
    - sim_threshold     (prolongation detection)

  For each new audio clip (one "task"):
    1. Run `inner_steps` gradient-descent steps using finite-difference
       gradient estimation on the disfluency score.
    2. After inner steps, update meta-parameters toward the task-optimal
       parameters using the Reptile outer update:
           theta <- theta + meta_lr * (theta_task - theta)

  The disfluency score measures how much stuttering remains after
  correction (lower is better).
"""

import math
import copy
import json
import os
import numpy as np

from config import (MAML_INNER_LR, MAML_META_LR, MAML_INNER_STEPS,
                    ENERGY_THRESHOLD, MAX_PAUSE_S, SIM_THRESHOLD,
                    TARGET_SR, MAML_SAVE_PATH)


class ReptileMAML:
    """
    Step 10: Reptile MAML for adapting DSP thresholds per speaker.

    Attributes
    ----------
    params : dict — current meta-parameters
        - energy_threshold : float
        - max_pause_s      : float
        - sim_threshold    : float
    """

    def __init__(self,
                 meta_lr: float     = MAML_META_LR,
                 inner_lr: float    = MAML_INNER_LR,
                 inner_steps: int   = MAML_INNER_STEPS):
        self.meta_lr     = meta_lr
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps
        self.params = {
            "energy_threshold": ENERGY_THRESHOLD,
            "max_pause_s":      MAX_PAUSE_S,
            "sim_threshold":    SIM_THRESHOLD,
        }
        self.history = []   # List of disfluency scores across adapt() calls
        self.last_trace = []  # Per-step trace for report figures
        self._cached_ref_mfcc = None

    # ------------------------------------------------------------------ #

    def _mfcc_matrix(self, signal: np.ndarray, sr: int,
                     frame_ms: int = 50, hop_ms: int = 25, n_mfcc: int = 13) -> np.ndarray:
        from utils import compute_mfcc
        frame = max(1, int(sr * frame_ms / 1000))
        hop = max(1, int(sr * hop_ms / 1000))
        
        # Batch collect frames
        frames = [signal[s:s + frame] for s in range(0, len(signal) - frame + 1, hop)]
        if not frames:
            return np.zeros((1, n_mfcc), dtype=np.float32)
            
        # Vectors of MFCCs
        out = [compute_mfcc(f, sr=sr, n_mfcc=n_mfcc) for f in frames]
        return np.asarray(out, dtype=np.float32)

    def _disfluency_score(self, params: dict, signal: np.ndarray, sr: int) -> float:
        """
        Report-aligned objective:
          Score = exp( - mean( |MFCC_original - MFCC_processed| ) )
          Loss  = 1 - Score
        Lower loss = better correction.
        Imports here to avoid circular imports.
        """
        from segmentation import SpeechSegmenter
        from pause_corrector import PauseCorrector
        from prolongation_corrector import ProlongationCorrector
        from speech_reconstructor import SpeechReconstructor

        try:
            # Keep threshold fixed during MAML scoring so energy_threshold is actually learnable.
            seg = SpeechSegmenter(
                sr=sr,
                energy_threshold=params["energy_threshold"],
                auto_threshold=False,
            )
            frames, labels, _ = seg.segment(signal)
            speech_pct = labels.count("speech") / max(len(labels), 1) * 100.0
            if speech_pct < 5.0 or speech_pct > 98.0:
                seg = SpeechSegmenter(
                    sr=sr,
                    energy_threshold=params["energy_threshold"],
                    auto_threshold=True,
                )
                frames, labels, _ = seg.segment(signal)

            pc = PauseCorrector(sr=sr, max_pause_s=params["max_pause_s"])
            frames, labels, _ = pc.correct(frames, labels)

            prc = ProlongationCorrector(sr=sr, sim_threshold=params["sim_threshold"])
            frames, labels, _ = prc.correct(frames, labels)

            rec = SpeechReconstructor(sr=sr)
            processed = rec.reconstruct(frames, labels)
            if len(processed) < int(0.3 * sr):
                return 1.0

            ref = self._cached_ref_mfcc if self._cached_ref_mfcc is not None else self._mfcc_matrix(signal, sr)
            hyp = self._mfcc_matrix(processed, sr)
            n = min(len(ref), len(hyp))
            if n <= 0:
                return 1.0
            mad = float(np.mean(np.abs(ref[:n] - hyp[:n])))
            score = float(np.exp(-mad))
            loss = 1.0 - score
            return float(np.clip(loss, 0.0, 1.0))
        except Exception:
            return 1.0

    def _clamp(self, params: dict) -> dict:
        """Clamp all parameters to physically valid ranges."""
        p = copy.deepcopy(params)
        p["energy_threshold"] = float(np.clip(p["energy_threshold"], 1e-5, 0.05)) # Hard limit 5% to prevent full-file deletion on noisy audio
        p["max_pause_s"]      = max(p["max_pause_s"],      0.05)
        p["sim_threshold"]    = float(np.clip(p["sim_threshold"], 0.50, 0.999))
        return p

    # ------------------------------------------------------------------ #

    def adapt(self, signal: np.ndarray, sr: int = TARGET_SR) -> dict:
        """
        Run Reptile MAML on a single audio signal.
        Adapts `self.params` toward the optimal values for this speaker.

        Parameters
        ----------
        signal : np.ndarray — preprocessed audio
        sr     : int        — sample rate

        Returns
        -------
        adapted_params : dict — speaker-specific optimized thresholds
        """
        print("[ReptileMAML] Starting adaptive threshold optimization...")
        task_params = copy.deepcopy(self.params)
        keys = list(task_params.keys())
        best_params = copy.deepcopy(task_params)
        best_score = float("inf")
        self.last_trace = []
        self._cached_ref_mfcc = self._mfcc_matrix(signal, sr)

        for step in range(self.inner_steps):
            grad = {}
            for k in keys:
                delta = max(abs(task_params[k]) * 0.05, 1e-5)
                p_hi  = self._clamp({**task_params, k: task_params[k] + delta})
                p_lo  = self._clamp({**task_params, k: task_params[k] - delta})
                s_hi  = self._disfluency_score(p_hi, signal, sr)
                s_lo  = self._disfluency_score(p_lo, signal, sr)
                grad[k] = (s_hi - s_lo) / (2 * delta)

            # Inner gradient descent step
            for k in keys:
                task_params[k] -= self.inner_lr * grad[k]
            task_params = self._clamp(task_params)

            score = self._disfluency_score(task_params, signal, sr)
            self.history.append(score)
            if score < best_score:
                best_score = score
                best_params = copy.deepcopy(task_params)
            self.last_trace.append({
                "step": step + 1,
                "loss": float(score),
                "score": float(1.0 - score),
                "params": copy.deepcopy(task_params),
            })
            print(f"  [MAML step {step+1}/{self.inner_steps}] "
                  f"loss={score:.4f} | params={task_params}")

        # Reptile outer update towards the best inner-loop parameters.
        for k in keys:
            self.params[k] += self.meta_lr * (best_params[k] - self.params[k])
        self.params = self._clamp(self.params)

        print(f"[ReptileMAML] Best inner score: {best_score:.4f}")
        print(f"[ReptileMAML] Meta-params updated: {self.params}")
        self._cached_ref_mfcc = None
        return copy.deepcopy(self.params)

    # ------------------------------------------------------------------ #

    def save(self, path: str = MAML_SAVE_PATH):
        """Persist meta-parameters and history to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {"params": self.params, "history": self.history[-100:]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[ReptileMAML] Params saved -> {path}")

    def load(self, path: str = MAML_SAVE_PATH):
        """Load previously saved meta-parameters from JSON."""
        if not os.path.exists(path):
            print(f"[ReptileMAML] WARNING: No saved params at '{path}'. Using defaults.")
            return
        with open(path) as f:
            data = json.load(f)
        self.params  = data["params"]
        self.history = data.get("history", [])
        print(f"[ReptileMAML] Params loaded from {path}: {self.params}")

    # ------------------------------------------------------------------ #

    def reset(self):
        """Reset to default initial parameters."""
        self.params = {
            "energy_threshold": ENERGY_THRESHOLD,
            "max_pause_s":      MAX_PAUSE_S,
            "sim_threshold":    SIM_THRESHOLD,
        }
        self.history.clear()
        print("[ReptileMAML] Parameters reset to defaults.")
