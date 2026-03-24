import os
import json
import sys
import numpy as np
from pathlib import Path

print("DEBUG: Importing evaluation dependencies...")
from preprocessing import AudioPreprocessor
from segmentation import SpeechSegmenter
from prolongation_corrector import ProlongationCorrector
from repetition_corrector import RepetitionCorrector
print("DEBUG: Imports successful!")

# Constants for evaluation
FILES_TO_EVAL = [
    "M_0030_16y4m_1.wav",
    "M_0061_16y9m-1.wav",
    "M_0078_16y5m_1.wav",
    "M_0107_07y7m_1.wav",
    "M_1106_25y0m_1.wav",
]

ANNOTATIONS_DIR = Path("archive/intermediate")
AUDIO_DIR = Path("archive/audio")
OUTPUT_DIR = Path("results/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Threshold sweeps
REP_SIM_THRESHOLDS = [0.75, 0.80, 0.85, 0.90]
PROL_SIM_THRESHOLDS = [0.88, 0.90, 0.93, 0.95]

# Optional: relax multi-feature gates for evaluation sensitivity
PROL_FLUX_THRESHOLD = 0.05
PROL_FLATNESS_THRESHOLD = 0.50


def load_ground_truth(file_stem):
    txt_path = ANNOTATIONS_DIR / f"{file_stem}.txt"
    events = []
    if not txt_path.exists():
        print(f"DEBUG: Annotation file not found: {txt_path}")
        return events

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            timestamp = float(parts[0])
            tags = parts[2].replace('"', '').split(',')
            for tag in tags:
                events.append({
                    "time": timestamp,
                    "type": tag.strip()
                })
    return events


def compute_metrics(detected, ground_truth, type_map, target_type, window_s: float = 0.5):
    tp = 0
    fp = 0
    fn = 0

    gt_used = [False] * len(ground_truth)

    # Filter ground truth for the type we are interested in
    relevant_gt_indices = [j for j, gt in enumerate(ground_truth) if type_map.get(gt["type"]) == target_type]

    for det in detected:
        det_time = det.get("start_time", 0)
        match_found = False
        for j in relevant_gt_indices:
            if gt_used[j]:
                continue
            if abs(det_time - ground_truth[j]["time"]) <= window_s:
                tp += 1
                gt_used[j] = True
                match_found = True
                break
        if not match_found:
            fp += 1

    fn = sum(1 for j in relevant_gt_indices if not gt_used[j])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def run_detection_raw(audio_path: Path, rep_thr: float, prol_thr: float):
    """
    Run detection on the original (preprocessed) signal to keep timestamps
    aligned with TextGrid annotations. Avoids pause removal/reconstruction.
    """
    pre = AudioPreprocessor(target_sr=16000, noise_reduce=False, normalization_method="rms")
    signal, sr, _meta = pre.process(str(audio_path))

    seg = SpeechSegmenter(sr=sr, frame_ms=25, hop_ms=12, energy_threshold=0.01, auto_threshold=True)
    frames, labels, _energies = seg.segment(signal)

    prol = ProlongationCorrector(
        sr=sr,
        sim_threshold=prol_thr,
        hop_ms=12,
    )
    prol.flux_threshold = PROL_FLUX_THRESHOLD
    prol.flatness_threshold = PROL_FLATNESS_THRESHOLD
    _frames_out, _labels_out, prol_stats = prol.correct(frames, labels)

    rep = RepetitionCorrector(sr=sr, sim_threshold=rep_thr)
    _sig_out, rep_stats = rep.correct(signal)

    return {
        "sr": sr,
        "hop_ms": 12,
        "prolongations": prol_stats.get("detection_events", []),
        "repetitions": rep_stats.get("detection_events", []),
    }


def main():
    results = {}
    TYPE_MAP = {
        "sound_repetition": "repetition",
        "word_repetition": "repetition",
    }

    overall_metrics = {"repetition": []}
    per_thr_avg = {}

    # Sweep over repetition and prolongation thresholds in lockstep
    thresholds = list(zip(REP_SIM_THRESHOLDS, PROL_SIM_THRESHOLDS))

    for rep_thr, prol_thr in thresholds:
        label = f"rep={rep_thr:.2f}|prol={prol_thr:.2f}"
        print("\n" + "=" * 60)
        print(f"Evaluating with thresholds {label}")
        print("=" * 60)

        per_thr_overall = {"repetition": []}
        results[label] = {}

        for filename in FILES_TO_EVAL:
            try:
                print(f"\nEvaluating {filename}...")
                audio_path = AUDIO_DIR / filename
                if not audio_path.exists():
                    print(f"DEBUG: Audio file not found: {audio_path}")
                    continue

                file_stem = filename.replace('.wav', '').replace('.txt', '')
                gt_events = load_ground_truth(file_stem)
                print(f"DEBUG: Loaded {len(gt_events)} ground truth events (full file).")

                print("DEBUG: Running raw-signal detection (timeline aligned)...")
                det = run_detection_raw(audio_path, rep_thr=rep_thr, prol_thr=prol_thr)
                hop_s = det["hop_ms"] / 1000.0

                sys_repetitions = [
                    {"start_time": e["start_sample"] / float(det["sr"]), "system_type": "repetition"}
                    for e in det.get("repetitions", [])
                ]

                r_metrics = compute_metrics(sys_repetitions, gt_events, TYPE_MAP, "repetition", window_s=0.5)

                results[label][filename] = {
                    "repetition": r_metrics,
                    "raw_counts": {
                        "gt_total": len(gt_events),
                        "det_rep": len(sys_repetitions)
                    }
                }

                per_thr_overall["repetition"].append(r_metrics)

                print(f"  Repetition:   F1={r_metrics['f1']:.2f} (P={r_metrics['precision']:.2f}, R={r_metrics['recall']:.2f})")

            except Exception as e:
                print(f"ERROR: Failed to process {filename}: {e}")
                import traceback
                traceback.print_exc()

        overall_metrics["repetition"].extend(per_thr_overall["repetition"])

        per_thr_avg[label] = {}
        for t in ["repetition"]:
            if per_thr_overall[t]:
                per_thr_avg[label][t] = {
                    "precision": float(np.mean([m["precision"] for m in per_thr_overall[t]])),
                    "recall": float(np.mean([m["recall"] for m in per_thr_overall[t]])),
                    "f1": float(np.mean([m["f1"] for m in per_thr_overall[t]])),
                }

    # Save summary
    summary_path = OUTPUT_DIR / "uclass_eval_summary.json"
    print(f"\nDEBUG: Saving summary to {summary_path}...")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PER-THRESHOLD AVERAGES")
    print("=" * 60)
    print(f"{'Thresholds':<20} | {'Type':<12} | {'Precision':<9} | {'Recall':<6} | {'F1':<6}")
    print("-" * 60)
    best_rep = {"thr": None, "f1": -1.0}
    for rep_thr, prol_thr in thresholds:
        thr_key = f"rep={rep_thr:.2f}|prol={prol_thr:.2f}"
        for t in ["repetition"]:
            if t in per_thr_avg.get(thr_key, {}):
                m = per_thr_avg[thr_key][t]
                print(f"{thr_key:<20} | {t:<12} | {m['precision']:<9.2f} | {m['recall']:<6.2f} | {m['f1']:<6.2f}")
                if t == "repetition" and m["f1"] > best_rep["f1"]:
                    best_rep = {"thr": thr_key, "f1": m["f1"]}

    if best_rep["thr"] is not None:
        print(f"\nBest repetition F1 = {best_rep['f1']:.3f} at thresholds = {best_rep['thr']}")

    print("\n" + "=" * 50)
    print("FINAL EVALUATION TABLE (Repetition Only)")
    print("=" * 50)
    print(f"{'Type':<15} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 55)
    if overall_metrics["repetition"]:
        avg_p = np.mean([m["precision"] for m in overall_metrics["repetition"]])
        avg_r = np.mean([m["recall"] for m in overall_metrics["repetition"]])
        avg_f1 = np.mean([m["f1"] for m in overall_metrics["repetition"]])
        print(f"{'Repetition':<15} | {avg_p:<10.2f} | {avg_r:<10.2f} | {avg_f1:<10.2f}")


if __name__ == "__main__":
    main()
