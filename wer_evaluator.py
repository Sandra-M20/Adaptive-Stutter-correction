"""
wer_evaluator.py
================
Enhancement 2: Word Error Rate (WER) Evaluator

Proves mathematically that our stutter correction pipeline improves
Speech-to-Text (STT) accuracy.

Workflow:
  1. Take a dysfluent audio clip and its ground-truth transcript.
  2. Run STT on the ORIGINAL (uncorrected) audio → compute WER_before.
  3. Run our full correction pipeline on the audio.
  4. Run STT on the CORRECTED audio → compute WER_after.
  5. Report: Improvement = WER_before - WER_after.

WER Formula:
  WER = (Substitutions + Deletions + Insertions) / N_ref_words
  A WER of 0.0 = perfect transcription.
  A WER of 1.0 = completely wrong transcription.
"""

import os
import json
import numpy as np
import soundfile as sf

from metrics import wer, duration_reduction, fluency_ratio, prolongation_rate
from config import TARGET_SR, RESULTS_DIR


class WERResult:
    """Container for WER evaluation results."""
    def __init__(self, clip_path, ground_truth, original_text,
                 corrected_text, wer_before, wer_after):
        self.clip_path      = clip_path
        self.ground_truth   = ground_truth
        self.original_text  = original_text
        self.corrected_text = corrected_text
        self.wer_before     = wer_before
        self.wer_after      = wer_after
        self.improvement    = wer_before - wer_after

    def to_dict(self):
        return {
            "clip":           self.clip_path,
            "ground_truth":   self.ground_truth,
            "original_text":  self.original_text,
            "corrected_text": self.corrected_text,
            "wer_before":     round(self.wer_before, 4),
            "wer_after":      round(self.wer_after, 4),
            "improvement":    round(self.improvement, 4),
        }

    def __repr__(self):
        return (f"WERResult: before={self.wer_before:.3f} "
                f"after={self.wer_after:.3f} "
                f"improvement={self.improvement:+.3f}")


class WEREvaluator:
    """
    Enhancement 2: Evaluate WER before and after stutter correction.

    Parameters
    ----------
    use_whisper   : bool — use Whisper for transcription
    save_results  : bool — save results to RESULTS_DIR/wer_results.json
    """

    def __init__(self, use_whisper: bool = True,
                 save_results: bool = True):
        self.use_whisper  = use_whisper
        self.save_results = save_results
        self._stt         = None

    # ------------------------------------------------------------------ #

    def _get_stt(self):
        if self._stt is None:
            from speech_to_text import SpeechToText
            self._stt = SpeechToText()
        return self._stt

    def _transcribe(self, signal: np.ndarray, sr: int) -> str:
        if self.use_whisper:
            return self._get_stt().transcribe(signal, sr)
        # Fallback: word count approximation (no Whisper)
        n_frames = len(signal) // int(sr * 0.05)
        return " ".join([f"word{i}" for i in range(max(1, n_frames // 10))])

    # ------------------------------------------------------------------ #

    def evaluate_clip(self, audio_path: str,
                      ground_truth: str = "") -> WERResult:
        """
        Evaluate a single clip: compute WER before and after correction.

        Parameters
        ----------
        audio_path    : str — path to the (dysfluent) WAV clip
        ground_truth  : str — reference transcript (optional for relative comparison)

        Returns
        -------
        WERResult
        """
        from preprocessing import AudioPreprocessor
        from segmentation import SpeechSegmenter
        from pause_corrector import PauseCorrector
        from prolongation_corrector import ProlongationCorrector
        from speech_reconstructor import SpeechReconstructor

        print(f"\n[WER] Evaluating: {os.path.basename(audio_path)}")

        # Load audio
        signal, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if signal.ndim == 2:
            signal = signal.mean(axis=1)

        # 1. Transcribe ORIGINAL
        original_text = self._transcribe(signal, sr)
        wer_b = wer(ground_truth, original_text) if ground_truth else 0.0
        print(f"  [WER] Original   : '{original_text[:60]}...' | WER={wer_b:.3f}")

        # 2. Correct
        proc = AudioPreprocessor(target_sr=sr)
        clean, _ = proc.process(audio_path)

        seg = SpeechSegmenter(sr=sr)
        frames, labels, _ = seg.segment(clean)

        pc = PauseCorrector(sr=sr)
        frames, labels, _ = pc.correct(frames, labels)

        prc = ProlongationCorrector(sr=sr)
        frames, labels, _ = prc.correct(frames, labels)

        rec = SpeechReconstructor(sr=sr)
        corrected = rec.reconstruct(frames, labels)

        # 3. Transcribe CORRECTED
        corrected_text = self._transcribe(corrected, sr)
        wer_a = wer(ground_truth, corrected_text) if ground_truth else 0.0
        print(f"  [WER] Corrected  : '{corrected_text[:60]}...' | WER={wer_a:.3f}")
        print(f"  [WER] Improvement: {wer_b - wer_a:+.3f}")

        return WERResult(audio_path, ground_truth, original_text,
                         corrected_text, wer_b, wer_a)

    # ------------------------------------------------------------------ #

    def evaluate_batch(self, samples: list) -> dict:
        """
        Evaluate WER on a list of sample dicts from DatasetLoader.
        Each dict must have 'path' key. Returns aggregate report.
        """
        results  = []
        for s in samples:
            gt = s.get("transcript", "")
            r  = self.evaluate_clip(s["path"], gt)
            results.append(r)

        wer_before_list = [r.wer_before   for r in results if r.ground_truth]
        wer_after_list  = [r.wer_after    for r in results if r.ground_truth]
        improv_list     = [r.improvement  for r in results if r.ground_truth]

        import statistics
        report = {
            "n_clips":            len(results),
            "mean_wer_before":    statistics.mean(wer_before_list) if wer_before_list else None,
            "mean_wer_after":     statistics.mean(wer_after_list)  if wer_after_list  else None,
            "mean_improvement":   statistics.mean(improv_list)     if improv_list     else None,
            "clips":              [r.to_dict() for r in results],
        }

        if self.save_results:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            out_path = os.path.join(RESULTS_DIR, "wer_results.json")
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n[WER] Results saved -> {out_path}")

        if report["mean_improvement"] is not None:
            print(f"\n[WER] === Summary ===")
            print(f"      Clips evaluated : {report['n_clips']}")
            print(f"      Avg WER before  : {report['mean_wer_before']:.3f}")
            print(f"      Avg WER after   : {report['mean_wer_after']:.3f}")
            print(f"      Avg improvement : {report['mean_improvement']:+.3f}")

        return report
