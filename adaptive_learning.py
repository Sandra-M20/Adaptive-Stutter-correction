"""
adaptive_learning.py
====================
Adaptive parameter optimization (Reptile-style finite-difference updates)
for DSP stutter correction thresholds.

Optimized parameters:
- energy_threshold
- pause_threshold_s
- correlation_threshold

Objective:
Score = exp( -mean(|MFCC_original - MFCC_processed|) )
Loss  = 1 - Score
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import copy
import numpy as np

from utils import compute_mfcc


class AdaptiveReptileLearner:
    def __init__(
        self,
        inner_lr: float = 0.05,
        meta_lr: float = 0.10,
        iterations: int = 10,
    ):
        self.inner_lr = float(inner_lr)
        self.meta_lr = float(meta_lr)
        self.iterations = int(iterations)
        self.default_params = {
            "energy_threshold": 0.01,
            "noise_threshold": 0.01,
            "pause_threshold_s": 0.5,
            "correlation_threshold": 0.93,
        }

    def _mfcc_matrix(self, signal: np.ndarray, sr: int, frame_ms: int = 25, hop_ms: int = 10, n_mfcc: int = 13) -> np.ndarray:
        frame = max(1, int(sr * frame_ms / 1000))
        hop = max(1, int(sr * hop_ms / 1000))
        rows = []
        for i in range(0, len(signal) - frame + 1, hop):
            rows.append(compute_mfcc(signal[i:i + frame], sr=sr, n_mfcc=n_mfcc))
        if not rows:
            return np.zeros((1, n_mfcc), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _score(self, ref: np.ndarray, hyp: np.ndarray) -> Tuple[float, float]:
        n = min(len(ref), len(hyp))
        if n <= 0:
            return 0.0, 1.0
        mad = float(np.mean(np.abs(ref[:n] - hyp[:n])))
        score = float(np.exp(-mad))
        loss = float(np.clip(1.0 - score, 0.0, 1.0))
        return score, loss

    def _clamp(self, p: Dict[str, float]) -> Dict[str, float]:
        q = copy.deepcopy(p)
        q["energy_threshold"] = float(np.clip(q["energy_threshold"], 0.001, 0.05))
        q["noise_threshold"] = float(np.clip(q["noise_threshold"], 0.001, 0.05))
        q["pause_threshold_s"] = float(np.clip(q["pause_threshold_s"], 0.20, 1.20))
        q["correlation_threshold"] = float(np.clip(q["correlation_threshold"], 0.80, 0.99))
        return q

    def optimize(
        self,
        signal: np.ndarray,
        sr: int,
        dsp_runner: Callable[[np.ndarray, int, Dict[str, float]], np.ndarray],
        initial_params: Dict[str, float] | None = None,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        `dsp_runner` must return processed audio for provided params.
        """
        params = self._clamp(initial_params or self.default_params)
        task_params = copy.deepcopy(params)
        best_params = copy.deepcopy(task_params)
        ref_mfcc = self._mfcc_matrix(signal, sr)
        best_loss = 1.0
        logs: List[Dict[str, float]] = []
        keys = list(task_params.keys())

        for step in range(1, self.iterations + 1):
            grads: Dict[str, float] = {}
            for k in keys:
                delta = max(abs(task_params[k]) * 0.05, 1e-4)
                p_hi = self._clamp({**task_params, k: task_params[k] + delta})
                p_lo = self._clamp({**task_params, k: task_params[k] - delta})
                y_hi = dsp_runner(signal, sr, p_hi)
                y_lo = dsp_runner(signal, sr, p_lo)
                s_hi, l_hi = self._score(ref_mfcc, self._mfcc_matrix(y_hi, sr))
                s_lo, l_lo = self._score(ref_mfcc, self._mfcc_matrix(y_lo, sr))
                grads[k] = (l_hi - l_lo) / (2.0 * delta)

            for k in keys:
                task_params[k] = float(task_params[k] - self.inner_lr * grads[k])
            task_params = self._clamp(task_params)

            y = dsp_runner(signal, sr, task_params)
            score, loss = self._score(ref_mfcc, self._mfcc_matrix(y, sr))
            if loss < best_loss:
                best_loss = loss
                best_params = copy.deepcopy(task_params)

            logs.append(
                {
                    "iteration": step,
                    "score": float(score),
                    "loss": float(loss),
                    "energy_threshold": float(task_params["energy_threshold"]),
                    "noise_threshold": float(task_params["noise_threshold"]),
                    "pause_threshold_s": float(task_params["pause_threshold_s"]),
                    "correlation_threshold": float(task_params["correlation_threshold"]),
                }
            )

            # stabilization/oscillation stop
            if len(logs) >= 6:
                tail = [x["loss"] for x in logs[-4:]]
                if float(np.std(tail)) < 5e-4:
                    break

        # Reptile meta update toward best task params.
        out = {}
        for k in keys:
            out[k] = float(params[k] + self.meta_lr * (best_params[k] - params[k]))
        out = self._clamp(out)
        return out, logs
