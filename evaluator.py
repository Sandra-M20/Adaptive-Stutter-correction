"""
evaluator.py
============
System-Level Performance Evaluator

Runs the complete 13-step pipeline on a batch of clips from the UCLASS
archive and computes holistic performance metrics:

  - Duration Reduction  : How much shorter is the corrected audio?
  - Pause Removal Rate  : How many long pauses were eliminated?
  - Prolongation Rate   : How many prolongation events were removed?
  - Fluency Ratio       : % of output that is speech (vs. silence)
  - Disfluency Score    : Composite lower-is-better score

Results are exported to results/evaluation_report.json and printed
in a human-readable table.
"""

import os
import json
import time
import numpy as np
import soundfile as sf

from config import TARGET_SR, RESULTS_DIR
from metrics import (duration_reduction, fluency_ratio,
                     prolongation_rate, disfluency_score)


class SystemEvaluator:
    """
    Evaluate the full correction pipeline on archive clips.

    Parameters
    ----------
    use_adaptive : bool — enable Reptile MAML adaptive optimization
    max_clips    : int  — maximum number of clips to evaluate
    """

    def __init__(self, use_adaptive: bool = True, max_clips: int = 50):
        self.use_adaptive = use_adaptive
        self.max_clips    = max_clips
        os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #

    def _run_pipeline(self, signal: np.ndarray, sr: int) -> tuple:
        """Run the 13-step pipeline and return (corrected, stats_dict)."""
        from preprocessing import AudioPreprocessor
        from adaptive_optimizer import ReptileMAML
        from segmentation import SpeechSegmenter
        from pause_corrector import PauseCorrector
        from prolongation_corrector import ProlongationCorrector
        from speech_reconstructor import SpeechReconstructor

        proc = AudioPreprocessor(target_sr=sr)
        clean, _ = proc.process((signal, sr))

        params = {}
        if self.use_adaptive:
            maml = ReptileMAML()
            params = maml.adapt(clean, sr)

        seg = SpeechSegmenter(sr=sr,
                              energy_threshold=params.get("energy_threshold", 0.01))
        frames, labels, energies = seg.segment(clean)

        pc = PauseCorrector(sr=sr,
                            max_pause_s=params.get("max_pause_s", 0.5))
        frames, labels, pc_stats = pc.correct(frames, labels)

        prc = ProlongationCorrector(sr=sr,
                                   sim_threshold=params.get("sim_threshold", 0.96))
        frames, labels, prc_stats = prc.correct(frames, labels)

        rec = SpeechReconstructor(sr=sr)
        corrected = rec.reconstruct(frames, labels)

        stats = {
            "pauses_removed":       pc_stats["pauses_found"],
            "prolongations_removed": prc_stats["prolongation_events"],
        }
        return corrected, stats

    # ------------------------------------------------------------------ #

    def evaluate(self, samples: list = None) -> dict:
        """
        Run evaluation on a list of sample dicts (from DatasetLoader)
        or on the full archive if samples=None.

        Returns
        -------
        report : dict — aggregated evaluation metrics
        """
        if samples is None:
            from dataset_loader import DatasetLoader
            loader  = DatasetLoader()
            samples = loader.sample_batch(self.max_clips, dysfluent_only=True, seed=42)

        print(f"\n[Evaluator] Evaluating {len(samples)} clips...")
        results = []
        t0      = time.time()

        for i, sample in enumerate(samples):
            path = sample["path"]
            try:
                signal, sr = sf.read(path, dtype="float32", always_2d=False)
                if signal.ndim == 2:
                    signal = signal.mean(axis=1)

                corrected, stats = self._run_pipeline(signal, sr)

                dr   = duration_reduction(signal, corrected)
                fr   = fluency_ratio(corrected, sr)
                pr   = prolongation_rate(corrected, sr)
                ds   = disfluency_score(corrected, sr)

                result = {
                    "clip":                   os.path.basename(path),
                    "duration_reduction_pct": round(dr, 2),
                    "fluency_ratio_pct":      round(fr, 2),
                    "prolongation_rate":      round(pr, 4),
                    "disfluency_score":       round(ds, 4),
                    **stats,
                }
                results.append(result)
                print(f"  [{i+1:3d}/{len(samples)}] {os.path.basename(path):35s} "
                      f"| DR={dr:+.1f}% FR={fr:.1f}% DS={ds:.3f}")
            except Exception as e:
                print(f"  [{i+1}] ERROR on {path}: {e}")

        elapsed = time.time() - t0

        # Aggregate
        def avg(key): return sum(r[key] for r in results) / max(len(results), 1)

        report = {
            "n_clips_evaluated":         len(results),
            "evaluation_time_s":         round(elapsed, 2),
            "mean_duration_reduction_pct": round(avg("duration_reduction_pct"), 2),
            "mean_fluency_ratio_pct":    round(avg("fluency_ratio_pct"), 2),
            "mean_prolongation_rate":    round(avg("prolongation_rate"), 4),
            "mean_disfluency_score":     round(avg("disfluency_score"), 4),
            "total_pauses_removed":      sum(r["pauses_removed"] for r in results),
            "total_prolongations_removed": sum(r["prolongations_removed"] for r in results),
            "clips":                     results,
        }

        # Save
        out = os.path.join(RESULTS_DIR, "evaluation_report.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n[Evaluator] === Results ===")
        print(f"  Clips evaluated     : {report['n_clips_evaluated']}")
        print(f"  Evaluation time     : {elapsed:.1f}s")
        print(f"  Mean duration reduc : {report['mean_duration_reduction_pct']:+.1f}%")
        print(f"  Mean fluency ratio  : {report['mean_fluency_ratio_pct']:.1f}%")
        print(f"  Disfluency score    : {report['mean_disfluency_score']:.4f}")
        print(f"  Total pauses removed: {report['total_pauses_removed']}")
        print(f"  Total prolong. rem. : {report['total_prolongations_removed']}")
        print(f"  Report saved        : {out}")

        return report
