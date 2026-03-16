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
import logging
import numpy as np
import soundfile as sf
import logging

def _calculate_snr_improvement(original: np.ndarray, corrected: np.ndarray) -> float:
    """Calculate SNR improvement in dB"""
    def snr(sig):
        power = np.mean(sig ** 2)
        noise = np.var(sig - np.mean(sig))
        return 10 * np.log10(power / (noise + 1e-10))
    
    return round(snr(corrected) - snr(original), 2)

def _calculate_lsd(original: np.ndarray, corrected: np.ndarray) -> float:
    """Calculate Log Spectral Distance"""
    min_len = min(len(original), len(corrected))
    s1 = np.log(np.abs(np.fft.rfft(original[:min_len])) + 1e-10)
    s2 = np.log(np.abs(np.fft.rfft(corrected[:min_len])) + 1e-10)
    return round(float(np.sqrt(np.mean((s1 - s2) ** 2))), 4)

from audio_input import AudioInputManager
from preprocessing import AudioPreprocessor
from segmentation import SpeechSegmenter
from reconstruction.reconstructor import Reconstructor
from speech_reconstructor import SpeechReconstructor
from correction.pause_corrector import PauseCorrector
from correction.prolongation_corrector import ProlongationCorrector
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
    transcript_orig: str = ""


class AdaptiveStutterPipeline:
    def __init__(
        self,
        target_sr: int = 16000,
        frame_ms: int = 25,
        hop_ms: int = 12,
        max_total_reduction: float = 0.40,
        use_enhancer: bool = False,
        output_gain_db: float = 8.0,
        transcribe: bool = False,
        use_silent_stutter: bool = True,
        use_repetition: bool = True,
        use_report_corr14: bool = False,
        onset_guard_s: float = 4.0,
        mode: str = "professional",
        noise_reduce: bool = True,  # Add noise reduction control
    ):
        """If use_report_corr14=True the prolongation detector uses the report-style
        correlation score threshold (corr_score = (sim+1)*10 >= 14) rather than
        raw cosine similarity.  Default is False (cosine-similarity mode is
        more robust for real speech).
        
        mode: 'professional' (default) or 'paper' (original dissertation logic).
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
        self.onset_guard_s = float(onset_guard_s)
        self.mode = mode.lower()
        self.input_manager = AudioInputManager(target_sr=target_sr)
        self.preprocessor = AudioPreprocessor(target_sr=target_sr, noise_reduce=noise_reduce, normalization_method='rms')
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
        if getattr(self, "mode", "professional") == "paper":
            return self._run_paper_dsp(signal, sr, params)
        
        # Original professional pipeline logic
        seg_thr = float(max(params["energy_threshold"], params.get("noise_threshold", params["energy_threshold"])))
        seg = SpeechSegmenter(
            sr=sr,
            frame_ms=self.frame_ms,
            hop_ms=self.hop_ms,
            energy_threshold=seg_thr,
        )
        frames, labels, _ = seg.segment(signal)
        speech_pct = labels.count("speech") / max(len(labels), 1) * 100.0
        if speech_pct < 5.0 or speech_pct > 98.0:
            seg = SpeechSegmenter(
                sr=sr,
                frame_ms=self.frame_ms,
                hop_ms=self.hop_ms,
                energy_threshold=seg_thr,
            )
            frames, labels, _ = seg.segment(signal)
            
        # Implementation of "skip-first-N-seconds" guard
        if self.onset_guard_s > 0:
            guard_frames = int(self.onset_guard_s * 1000 / self.hop_ms)
            for i in range(min(guard_frames, len(labels))):
                labels[i] = "speech" # Protect initial frames from being marked as silence or disfluency

        # Requested controller order: pause correction then prolongation correction.
        pause = LongPauseRemover(
            sr=sr,
            frame_ms=self.frame_ms,
            hop_ms=self.hop_ms,
            pause_threshold_s=float(params["pause_threshold_s"]),
        )
        frames, labels, pause_stats = pause.process(frames, labels)
        
        # Debug logging to identify stat key names
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Pause stats raw: {pause_stats}")
        
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
        original_len = len(signal)
        corrected_len = len(corrected)
        actual_reduction = (1.0 - corrected_len / original_len) * 100
        
        logger.info(f"Duration reduction: {actual_reduction:.1f}% (max allowed: {self.max_total_reduction*100:.1f}%)")
        
        min_len = int(original_len * (1.0 - self.max_total_reduction))
        if corrected_len < min_len:
            logger.warning(f"Safety gate triggered! corrected={corrected_len} < min={min_len}, reverting to original")
            corrected = signal.copy()
        elif self.use_enhancer:
            corrected = self.enhancer.enhance(corrected)

        rep_stats = {"repetition_events": 0, "detection_events": []}
        if self.use_repetition and len(corrected) > int(0.8 * sr):
            if len(corrected) / max(sr, 1) <= 45.0:
                corrected, rep_stats = self.rep_short.correct(corrected)
            else:
                corrected, rep_stats = self.rep_long.correct(corrected)

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
            "repetitions_removed": float(rep_stats.get("repetition_events", 0)),
            "duration_reduction_pct": (1.0 - len(corrected) / max(len(signal), 1)) * 100.0,
            "snr_improvement_db": _calculate_snr_improvement(signal, corrected),
            "log_spectral_distance": _calculate_lsd(signal, corrected),
            "detection_events": {
                "pauses": pause_stats.get("detection_events", []),
                "prolongations": prol_stats.get("detection_events", []),
                "repetitions": rep_stats.get("detection_events", []),
            }
        }
        
        # Debug logging for final stats
        logger.info(f"Final stats keys: {list(stats.keys())}")
        logger.info(f"Pause stats keys: {list(pause_stats.keys())}")
        
        return corrected.astype(np.float32), stats

    def _run_paper_dsp(self, signal: np.ndarray, sr: int, params: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Implementation of the exact DSP pipeline from the paper/notebook."""
        from utils import compute_lpc, cosine_similarity, compute_mfcc
        
        streak_th = float(params.get("streak_threshold", params.get("correlation_threshold", 14)))
        noise_rms = float(params.get("noise_threshold", params.get("energy_threshold", 0.015)))
        corr_th   = float(params.get("corr_threshold", 0.92))
        
        frame_size = int(sr * 25 / 1000)
        hop_size   = int(sr * 10 / 1000)
        
        # 1. Split into clips (RMS-based VAD from notebook)
        n = len(signal)
        nf = (n - frame_size) // hop_size + 1
        rms = np.array([np.sqrt(np.mean(signal[i*hop_size : i*hop_size + frame_size]**2)) for i in range(nf)])
        is_speech = rms > noise_rms
        mask = np.zeros(n, dtype=bool)
        for i, sp in enumerate(is_speech):
            mask[i*hop_size : min(i*hop_size + hop_size, n)] = sp
            
        intervals = []
        in_sp = False; seg_s = 0
        min_samp = int(sr * 0.05)
        for i in range(n):
            if mask[i] and not in_sp: in_sp = True; seg_s = i
            elif not mask[i] and in_sp:
                in_sp = False
                if i - seg_s >= min_samp: intervals.append((seg_s, i))
        if in_sp: intervals.append((seg_s, n))
        
        clips_data = []; gaps = []; prev = 0
        for s, e in intervals:
            gaps.append((prev, s))
            clips_data.append(signal[s:e])
            prev = e
        gaps.append((prev, n))
        
        # 2. Process clips
        parts = []
        prev_mfcc = None
        prolong_events = 0
        repetition_events = 0
        frames_removed = 0
        
        for idx, c_audio in enumerate(clips_data):
            # Gap trimming (Section 6.2)
            gap_audio = signal[gaps[idx][0]:gaps[idx][1]]
            max_p_samp = int(0.5 * sr)
            if len(gap_audio) > max_p_samp:
                parts.append(gap_audio[:max_p_samp])
            elif len(gap_audio) > 0:
                parts.append(gap_audio)
                
            # Too short
            if len(c_audio) < frame_size:
                parts.append(c_audio)
                continue
                
            # Repetition check (Chapter 4)
            curr_mfcc = compute_mfcc(c_audio[:frame_size], sr=sr) 
            if prev_mfcc is not None:
                dist = np.mean(np.abs(curr_mfcc - prev_mfcc))
                if dist < 3.5:
                    repetition_events += 1
                    continue
            
            # Prolongation removal (Section 6.1)
            cnf = (len(c_audio) - frame_size) // hop_size + 1
            cfeats = []
            for i in range(cnf):
                f = c_audio[i*hop_size : i*hop_size + frame_size]
                lpc = compute_lpc(f, order=12)
                en  = np.mean(f**2)
                zcr = np.mean(np.abs(np.diff(np.sign(f)))) / 2.0
                cfeats.append(np.concatenate([[en, zcr], lpc]))
            
            cretain = np.ones(len(c_audio), dtype=bool)
            streak = 0
            found_prolong = False
            for i in range(1, len(cfeats)):
                if cosine_similarity(cfeats[i-1], cfeats[i]) >= corr_th:
                    streak += 1
                    if streak > streak_th:
                        cretain[i*hop_size : min(i*hop_size + hop_size, len(c_audio))] = False
                        found_prolong = True
                else:
                    streak = 0
            
            corrected_clip = c_audio[cretain]
            parts.append(corrected_clip)
            prev_mfcc = compute_mfcc(corrected_clip[:frame_size], sr=sr)
            if found_prolong: prolong_events += 1
            frames_removed += (len(c_audio) - len(corrected_clip)) // hop_size

        # Final gap
        final_gap = signal[gaps[-1][0]:gaps[-1][1]]
        if len(final_gap) > int(0.5 * sr):
            parts.append(final_gap[:int(0.5 * sr)])
        elif len(final_gap) > 0:
            parts.append(final_gap)
            
        corrected = np.concatenate(parts) if parts else signal.copy()
        
        stats = {
            "speech_pct": len(intervals),
            "pauses_found": len(intervals),
            "pause_frames_removed": 0,
            "silent_stutters_removed": 0,
            "silent_stutter_frames_removed": 0,
            "prolongation_events": float(prolong_events),
            "prolong_frames_removed": float(frames_removed),
            "repetitions_removed": float(repetition_events),
            "duration_reduction_pct": (1.0 - len(corrected) / max(len(signal), 1)) * 100.0,
            "snr_improvement_db": _calculate_snr_improvement(signal, corrected),
            "log_spectral_distance": _calculate_lsd(signal, corrected),
            "detection_events": {"pauses":[], "prolongations":[], "repetitions":[]}
        }
        return corrected.astype(np.float32), stats

    def run(
        self,
        audio_input: str | Tuple[np.ndarray, int],
        output_path: str = "output/corrected_main_pipeline.wav",
        optimize: bool = True,
        initial_params: Dict[str, float] | None = None,
        language: str | None = None,
        noise_reduce: Optional[bool] = None,
        over_subtraction: Optional[float] = None,
        target_rms: Optional[float] = None,
    ) -> PipelineRunResult:
        t0 = time.time()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if isinstance(audio_input, tuple):
            raw_sig, raw_sr = audio_input
            raw_sig = raw_sig.astype(np.float32)
            if raw_sig.ndim == 2:
                raw_sig = raw_sig.mean(axis=1)
            signal, sr, metadata = self.preprocessor.process(
                (raw_sig, raw_sr),
                noise_reduce=noise_reduce,
                over_subtraction=over_subtraction,
                target_rms=target_rms
            )
        else:
            signal, sr, metadata = self.preprocessor.process(
                audio_input,
                noise_reduce=noise_reduce,
                over_subtraction=over_subtraction,
                target_rms=target_rms
            )

        params = initial_params or {
            "energy_threshold": 0.01,
            "noise_threshold": 0.01,
            "pause_threshold_s": 0.3,
            "correlation_threshold": 0.85,
            "max_remove_ratio": 0.40,
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
        transcript_orig = ""
        if self.stt is not None:
            transcript = self.stt.transcribe(corrected, sr=sr, language=language)
            transcript_orig = self.stt.transcribe(signal, sr=sr, language=language)

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
            transcript_orig=transcript_orig,
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
            signal, sr, _ = self.preprocessor.process((signal, sr))
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
    # _plot_optional(res)
    print("Completed.")
    print("Output:", res.output_path)
    print("Params:", res.params)
    print("Stats:", res.stats)
    if res.transcript:
        print("Transcript:", res.transcript)
