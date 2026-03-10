"""
main_pipeline.py
================
Reference DSP + Adaptive-learning implementation following the report
methodology for stutter correction.

Pipeline:
Speech Input -> Preprocessing -> DSP (Prolongation + Long Pause) ->
Adaptive Learning (Reptile-style) -> Corrected Audio + Logs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import os
import time
import numpy as np
import soundfile as sf

from audio_input import AudioInputManager
from preprocessing import AudioPreprocessor
from segmentation import SpeechSegmenter
from speech_reconstructor import SpeechReconstructor
from pause_removal import LongPauseRemover
from prolongation_removal import ProlongationRemover
from adaptive_learning import AdaptiveReptileLearner
from audio_enhancer import AudioEnhancer
from speech_to_text import SpeechToText
from silent_stutter_detector import SilentStutterDetector
from repetition_corrector import RepetitionCorrector


@dataclass
class PipelineRunResult:
    corrected_audio: np.ndarray
    sr: int
    params: Dict[str, float]
    iteration_logs: List[Dict[str, float]]
    stats: Dict[str, float]
    output_path: str
    transcript: str = ""


class AdaptiveStutterPipeline:
    def __init__(
        self,
        target_sr: int = 22050,
        frame_ms: int = 25,
        hop_ms: int = 12,
        max_total_reduction: float = 0.18,
        use_enhancer: bool = True,
        output_gain_db: float = 8.0,
        transcribe: bool = True,
        use_silent_stutter: bool = True,
        use_repetition: bool = True,
        use_report_corr14: bool = False,
    ):
        """If use_report_corr14=True the prolongation detector uses the report-style
        correlation score threshold (corr_score = (sim+1)*10 >= 14) rather than
        raw cosine similarity.  Default is False (cosine-similarity mode is
        more robust for real speech).
        """
        self.target_sr = target_sr
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.max_total_reduction = max_total_reduction
        self.use_enhancer = use_enhancer
        self.output_gain_db = float(output_gain_db)
        self.transcribe = bool(transcribe)
        self.use_silent_stutter = bool(use_silent_stutter)
        self.use_repetition = bool(use_repetition)
        self.use_report_corr14 = bool(use_report_corr14)
        self.input_manager = AudioInputManager(target_sr=target_sr)
        self.preprocessor = AudioPreprocessor(target_sr=target_sr, noise_reduce=True)
        self.learner = AdaptiveReptileLearner(iterations=10)
        self.enhancer = AudioEnhancer(sr=target_sr)
        self.stt = SpeechToText() if self.transcribe else None
        self.silent_detector = SilentStutterDetector(sr=target_sr, hop_ms=hop_ms)
        self.rep_short = RepetitionCorrector(
            sr=target_sr,
            chunk_ms=280,
            dtw_threshold=2.2,
            max_total_removal_ratio=0.04,
        )
        self.rep_long = RepetitionCorrector(
            sr=target_sr,
            chunk_ms=320,
            dtw_threshold=1.9,
            max_total_removal_ratio=0.015,
        )

    def _run_dsp(self, signal: np.ndarray, sr: int, params: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        # Energy/noise thresholds both represent speech-vs-silence gating.
        seg_thr = float(max(params["energy_threshold"], params.get("noise_threshold", params["energy_threshold"])))
        seg = SpeechSegmenter(
            sr=sr,
            frame_ms=self.frame_ms,
            hop_ms=self.hop_ms,
            energy_threshold=seg_thr,
            auto_threshold=False,
        )
        frames, labels, _ = seg.segment(signal)
        speech_pct = labels.count("speech") / max(len(labels), 1) * 100.0
        if speech_pct < 5.0 or speech_pct > 98.0:
            seg = SpeechSegmenter(
                sr=sr,
                frame_ms=self.frame_ms,
                hop_ms=self.hop_ms,
                energy_threshold=seg_thr,
                auto_threshold=True,
            )
            frames, labels, _ = seg.segment(signal)

        # Requested controller order: pause correction then prolongation correction.
        pause = LongPauseRemover(
            sr=sr,
            frame_ms=self.frame_ms,
            hop_ms=self.hop_ms,
            pause_threshold_s=float(params["pause_threshold_s"]),
        )
        frames, labels, pause_stats = pause.process(frames, labels)

        silent_stats = {"silent_stutters_removed": 0, "frames_removed": 0}
        if self.use_silent_stutter:
            frames, labels, silent_stats = self.silent_detector.correct(frames, labels)

        prol = ProlongationRemover(
            sr=sr,
            hop_ms=self.hop_ms,
            correlation_threshold=float(params["correlation_threshold"]),
            use_report_corr14=self.use_report_corr14,
        )
        frames, labels, prol_stats = prol.process(frames, labels)

        rec = SpeechReconstructor(sr=sr, frame_ms=self.frame_ms, hop_ms=self.hop_ms)
        corrected = rec.reconstruct(frames, labels)

        # prevent over-removal
        min_len = int(len(signal) * (1.0 - self.max_total_reduction))
        if len(corrected) < min_len:
            corrected = signal.copy()
        elif self.use_enhancer:
            corrected = self.enhancer.enhance(corrected)

        rep_removed = 0
        if self.use_repetition and len(corrected) > int(0.8 * sr):
            if len(corrected) / max(sr, 1) <= 45.0:
                corrected, rep_removed = self.rep_short.correct(corrected)
            else:
                corrected, rep_removed = self.rep_long.correct(corrected)

        # User-facing loudness boost with simple limiter.
        if self.output_gain_db > 0.0 and len(corrected) > 0:
            corrected = corrected * float(10 ** (self.output_gain_db / 20.0))
            peak = float(np.max(np.abs(corrected)) + 1e-12)
            if peak > 0.98:
                corrected = corrected * (0.98 / peak)

        stats = {
            "speech_pct": speech_pct,
            "pauses_found": float(pause_stats.get("pauses_found", 0)),
            "pause_frames_removed": float(pause_stats.get("frames_removed", 0)),
            "silent_stutters_removed": float(silent_stats.get("silent_stutters_removed", 0)),
            "silent_stutter_frames_removed": float(silent_stats.get("frames_removed", 0)),
            "prolongation_events": float(prol_stats.get("prolongation_events", 0)),
            "prolong_frames_removed": float(prol_stats.get("frames_removed", 0)),
            "repetitions_removed": float(rep_removed),
            "duration_reduction_pct": (1.0 - len(corrected) / max(len(signal), 1)) * 100.0,
        }
        return corrected.astype(np.float32), stats

    def run(
        self,
        audio_input: str | Tuple[np.ndarray, int],
        output_path: str = "output/corrected_main_pipeline.wav",
        optimize: bool = True,
        initial_params: Dict[str, float] | None = None,
        language: str | None = None,
    ) -> PipelineRunResult:
        t0 = time.time()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if isinstance(audio_input, tuple):
            raw_sig, raw_sr = audio_input
            raw_sig = raw_sig.astype(np.float32)
            if raw_sig.ndim == 2:
                raw_sig = raw_sig.mean(axis=1)
            signal, sr = self.preprocessor.process((raw_sig, raw_sr))
        else:
            signal, sr = self.preprocessor.process(audio_input)

        params = initial_params or {
            "energy_threshold": 0.01,
            "noise_threshold": 0.01,
            "pause_threshold_s": 0.5,
            "correlation_threshold": 0.93,
        }
        logs: List[Dict[str, float]] = []

        if optimize:
            params, logs = self.learner.optimize(
                signal=signal,
                sr=sr,
                dsp_runner=lambda x, y, p: self._run_dsp(x, y, p)[0],
                initial_params=params,
            )

        corrected, dsp_stats = self._run_dsp(signal, sr, params)
        sf.write(output_path, corrected, sr)
        transcript = ""
        if self.stt is not None:
            transcript = self.stt.transcribe(corrected, sr=sr, language=language)

        final_stats = {
            **dsp_stats,
            "input_duration_s": len(signal) / sr,
            "output_duration_s": len(corrected) / sr,
            "runtime_s": time.time() - t0,
        }
        return PipelineRunResult(
            corrected_audio=corrected,
            sr=sr,
            params=params,
            iteration_logs=logs,
            stats=final_stats,
            output_path=output_path,
            transcript=transcript,
        )

    def run_near_realtime(
        self,
        signal: np.ndarray,
        sr: int,
        chunk_s: float = 1.0,
        params: Dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        Near real-time mode: process fixed chunks sequentially with the same
        DSP parameters, then concatenate.
        """
        if signal.ndim == 2:
            signal = signal.mean(axis=1)
        if sr != self.target_sr:
            signal, sr = self.preprocessor.process((signal, sr))
        params = params or {
            "energy_threshold": 0.01,
            "noise_threshold": 0.01,
            "pause_threshold_s": 0.5,
            "correlation_threshold": 0.93,
        }
        chunk_n = max(1, int(chunk_s * sr))
        parts = []
        for s in range(0, len(signal), chunk_n):
            chunk = signal[s:s + chunk_n]
            if len(chunk) < int(0.2 * sr):
                parts.append(chunk)
                continue
            y, _ = self._run_dsp(chunk, sr, params)
            parts.append(y)
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

    @staticmethod
    def save_logs(result: PipelineRunResult, json_path: str = "results/main_pipeline_logs.json") -> None:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "params": result.params,
                    "stats": result.stats,
                    "iterations": result.iteration_logs,
                    "output_path": result.output_path,
                    "transcript": result.transcript,
                },
                f,
                indent=2,
            )


def _plot_optional(result: PipelineRunResult) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not result.iteration_logs:
        return
    os.makedirs("results", exist_ok=True)
    it = [x["iteration"] for x in result.iteration_logs]
    loss = [x["loss"] for x in result.iteration_logs]
    e = [x["energy_threshold"] for x in result.iteration_logs]
    n = [x.get("noise_threshold", x["energy_threshold"]) for x in result.iteration_logs]
    p = [x["pause_threshold_s"] for x in result.iteration_logs]
    c = [x["correlation_threshold"] for x in result.iteration_logs]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(it, loss, marker="o")
    ax.set_title("Adaptive Learning Loss (L = 1 - Score)")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/main_loss_curve.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(it, e, marker="o", label="energy_threshold")
    ax.plot(it, n, marker="o", label="noise_threshold")
    ax.plot(it, p, marker="o", label="pause_threshold_s")
    ax.plot(it, c, marker="o", label="correlation_threshold")
    ax.set_title("Threshold Evolution")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/main_threshold_evolution.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive DSP stutter correction pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input audio file")
    parser.add_argument("--output", type=str, default="output/corrected_main_pipeline.wav", help="Output wav")
    parser.add_argument("--no-opt", action="store_true", help="Disable adaptive learning")
    args = parser.parse_args()

    pipe = AdaptiveStutterPipeline()
    res = pipe.run(args.input, output_path=args.output, optimize=(not args.no_opt))
    pipe.save_logs(res)
    _plot_optional(res)
    print("Completed.")
    print("Output:", res.output_path)
    print("Params:", res.params)
    print("Stats:", res.stats)
    if res.transcript:
        print("Transcript:", res.transcript)
