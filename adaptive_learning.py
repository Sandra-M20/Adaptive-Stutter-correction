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
            "pause_threshold_s": 0.25,
            "correlation_threshold": 0.75,
            "max_remove_ratio": 0.40,
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
        if "energy_threshold" in q:
            q["energy_threshold"] = float(np.clip(q["energy_threshold"], 0.001, 0.05))
        if "noise_threshold" in q:
            q["noise_threshold"] = float(np.clip(q["noise_threshold"], 0.001, 0.05))
        if "pause_threshold_s" in q:
            q["pause_threshold_s"] = float(np.clip(q["pause_threshold_s"], 0.25, 1.00))
        if "correlation_threshold" in q:
            q["correlation_threshold"] = float(np.clip(q["correlation_threshold"], 0.70, 0.92))
        if "max_remove_ratio" in q:
            q["max_remove_ratio"] = float(np.clip(q["max_remove_ratio"], 0.10, 0.60))
        if "streak_threshold" in q:
            q["streak_threshold"] = float(np.clip(q["streak_threshold"], 3.0, 40.0))
        if "corr_threshold" in q:
            q["corr_threshold"] = float(np.clip(q["corr_threshold"], 0.50, 0.99))
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
        best_loss = float("inf")
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
                actual_delta = (p_hi[k] - p_lo[k]) / 2.0
                if abs(actual_delta) < 1e-8:
                    grads[k] = 0.0
                else:
                    grads[k] = (l_hi - l_lo) / (2.0 * actual_delta)

            for k in keys:
                task_params[k] = float(task_params[k] - self.inner_lr * grads[k])
                
            # Decaying exploration noise (Paper Mode enhancement)
            noise_scale = 1.0 / (1.0 + step * 0.8)
            for k in keys:
                delta = max(abs(task_params[k]) * 0.05, 1e-4)
                noise = np.random.normal(0, delta * noise_scale)
                task_params[k] += noise
                
            task_params = self._clamp(task_params)

            y = dsp_runner(signal, sr, task_params)
            score, loss = self._score(ref_mfcc, self._mfcc_matrix(y, sr))
            if loss < best_loss:
                best_loss = loss
                best_params = copy.deepcopy(task_params)

            log_entry = {
                "iteration": step,
                "score": float(score),
                "loss": float(loss),
            }
            log_entry.update({k: float(v) for k, v in task_params.items()})
            logs.append(log_entry)

            # stabilization/oscillation stop
            if len(logs) >= 4:
                tail = [x["loss"] for x in logs[-4:]]
                if float(np.std(tail)) < 5e-4:
                    break

        # Reptile meta update toward best task params.
        out = {}
        for k in best_params.keys():
            meta_val = params.get(k, best_params[k])
            out[k] = float(meta_val + self.meta_lr * (best_params[k] - meta_val))
        out = self._clamp(out)
        return out, logs

    def load_speaker_calibration(self, speaker_id: str, calibration_dir: str = "maml_calibration") -> np.ndarray:
        """
        Load and concatenate calibration clips for a specific speaker.
        Used to run per-speaker Reptile optimization before processing.
        Returns concatenated audio signal as np.ndarray at self target sr.
        """
        import os
        import soundfile as sf
        
        speaker_path = os.path.join(calibration_dir, speaker_id)
        if not os.path.exists(speaker_path):
            print(f"[Warning] Calibration directory not found: {speaker_path}")
            return None
        
        # Get all WAV files in the speaker directory
        wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
        if not wav_files:
            print(f"[Warning] No WAV files found in: {speaker_path}")
            return None
        
        # Load and concatenate all calibration clips
        audio_segments = []
        for wav_file in sorted(wav_files):
            file_path = os.path.join(speaker_path, wav_file)
            try:
                audio, sr = sf.read(file_path)
                # Convert to mono if needed
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                # Resample to target rate if needed
                if sr != 16000:  # TARGET_SR
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                audio_segments.append(audio)
            except Exception as e:
                print(f"[Warning] Failed to load {wav_file}: {e}")
                continue
        
        if not audio_segments:
            print(f"[Warning] No valid audio segments loaded for speaker {speaker_id}")
            return None
        
        # Concatenate all segments
        concatenated_audio = np.concatenate(audio_segments)
        print(f"[Calibration] Loaded {len(audio_segments)} clips for speaker {speaker_id}, total duration: {len(concatenated_audio)/16000:.2f}s")
        
        return concatenated_audio
