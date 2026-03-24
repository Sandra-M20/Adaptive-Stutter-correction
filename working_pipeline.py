"""
working_pipeline.py
===================
A clean, self-contained stutter correction pipeline using only the
real, working modules at the project root.

Steps:
  1. Resample to TARGET_SR
  2. Normalize amplitude
  3. Segment into speech/silence frames (SpeechSegmenter)
  4. Compress long pauses (PauseCorrector)
  5. Remove prolongations (ProlongationCorrector)
  6. Reconstruct signal from frames via Overlap-Add
  7. Remove repetitions (RepetitionCorrector)
  8. Final normalize
"""

import numpy as np
from config import TARGET_SR, FRAME_MS, HOP_MS


# ─────────────────────────────────────────────────────────────────────────────
# Overlap-Add reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct(frames: list, hop_size: int) -> np.ndarray:
    """
    Reconstruct audio from overlapping frames using Overlap-Add with a
    Hanning window.  Works correctly for 50 % overlap (hop = frame/2).
    """
    if not frames:
        return np.array([], dtype=np.float32)

    frame_size = len(frames[0])
    window = np.hanning(frame_size).astype(np.float64)

    # Hanning OLA normalisation constant (for 50 % overlap)
    # sum of squared Hanning windows at any point ≈ 0.5
    # We compute it exactly from two adjacent windows so it is accurate.
    norm = window ** 2
    if frame_size > hop_size:
        norm[:frame_size - hop_size] += window[hop_size:] ** 2

    out_len = hop_size * (len(frames) - 1) + frame_size
    out  = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)

    for i, frame in enumerate(frames):
        s = i * hop_size
        e = s + frame_size
        f = np.asarray(frame, dtype=np.float64)
        out[s:e]  += f * window
        wsum[s:e] += window ** 2

    mask = wsum > 1e-10
    out[mask] /= wsum[mask]
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive segmentation threshold
# ─────────────────────────────────────────────────────────────────────────────

def _compute_energy_threshold(signal: np.ndarray, frame_size: int, hop_size: int) -> float:
    """
    Pick a robust speech/silence threshold.

    Strategy:
      1. Compute Short-Time Energy for every frame.
      2. Split energies into a lower half and upper half at the median.
      3. The threshold is the midpoint between the medians of those two groups.
         This finds the natural gap between quiet (silence/noise) and loud
         (speech) frames — more robust than a fixed percentile on noisy audio.
    """
    energies = []
    for s in range(0, len(signal) - frame_size + 1, hop_size):
        frame = signal[s: s + frame_size]
        energies.append(float(np.mean(frame ** 2)))

    if not energies:
        return 0.01

    arr = np.array(energies)
    median = np.median(arr)
    low_med  = np.median(arr[arr <= median]) if np.any(arr <= median) else median
    high_med = np.median(arr[arr >  median]) if np.any(arr >  median) else median

    # Midpoint of the two clusters
    threshold = (low_med + high_med) / 2.0
    # Guard against degenerate signals
    threshold = float(np.clip(threshold, 1e-6, 0.5))
    print(f"[Pipeline] Energy threshold (kmeans-1D): {threshold:.6f}  "
          f"(low_med={low_med:.6f}, high_med={high_med:.6f})")
    return threshold


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(signal: np.ndarray, sr: int,
                 pause_threshold: float      = 0.60,   # 600ms before flagging a block
                 retain_ratio: float         = 0.25,
                 similarity_threshold: float = 0.92,   # 0.92 — only extremely stable frames qualify
                 min_prolong_frames: int     = 12) -> dict:  # 300ms minimum — real prolongations only
    """
    Run the full stutter correction pipeline.

    Returns dict with:
        corrected_signal, original_duration, corrected_duration,
        disfluency_removed (%), prolongation_events, pause_events,
        repetition_events, sr
    """
    from segmentation import SpeechSegmenter
    from pause_corrector import PauseCorrector
    from prolongation_corrector import ProlongationCorrector
    from repetition_corrector import RepetitionCorrector

    # ── Step 1: Resample ────────────────────────────────────────────────────
    if sr != TARGET_SR:
        try:
            import librosa
            signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)
        except Exception:
            from utils import resample
            signal = resample(signal, sr, TARGET_SR)
        sr = TARGET_SR

    signal = signal.astype(np.float32)
    original_duration = len(signal) / sr

    # ── Step 2: Normalize ───────────────────────────────────────────────────
    peak = np.max(np.abs(signal))
    if peak > 1e-8:
        signal = signal / peak

    frame_size = int(sr * FRAME_MS / 1000)
    hop_size   = int(sr * HOP_MS  / 1000)

    # ── Step 3: Segment (adaptive threshold) ─────────────────────────────────
    energy_thr = _compute_energy_threshold(signal, frame_size, hop_size)

    segmenter = SpeechSegmenter(
        sr=sr, frame_ms=FRAME_MS, hop_ms=HOP_MS,
        energy_threshold=energy_thr,
        auto_threshold=False,   # use our robust threshold instead
    )
    frames, labels, _ = segmenter.segment(signal)

    n_speech = labels.count("speech")
    print(f"[Pipeline] {n_speech}/{len(labels)} frames = speech "
          f"({100*n_speech/max(len(labels),1):.1f}%)")

    # ── Step 4: Pause correction ─────────────────────────────────────────────
    pause_corrector = PauseCorrector(
        sr=sr, frame_ms=FRAME_MS, hop_ms=HOP_MS,
        max_pause_s=pause_threshold,
        retain_ratio=retain_ratio,
    )
    frames, labels, pause_stats = pause_corrector.correct(frames, labels)

    # ── Step 5: Prolongation correction ──────────────────────────────────────
    prolong_corrector = ProlongationCorrector(
        sr=sr,
        sim_threshold=similarity_threshold,
        min_prolong_frames=min_prolong_frames,
        hop_ms=HOP_MS,
    )
    frames, labels, prolong_stats = prolong_corrector.correct(frames, labels)

    # ── Step 6: Overlap-Add reconstruction ───────────────────────────────────
    reconstructed = _reconstruct(frames, hop_size)

    if len(reconstructed) == 0:
        # Fallback: nothing to reconstruct — return original
        reconstructed = signal.copy()

    # ── Step 7: Repetition correction ────────────────────────────────────────
    rep_corrector = RepetitionCorrector(
        sr=sr,
        sim_threshold=0.94,           # raised to 0.94 — only true word/syllable repeats qualify
        max_total_removal_ratio=0.20,
    )
    corrected, rep_stats = rep_corrector.correct(reconstructed)

    # ── Step 8: Final normalize ───────────────────────────────────────────────
    if len(corrected) == 0:
        corrected = reconstructed.copy()

    peak2 = np.max(np.abs(corrected))
    if peak2 > 1e-8:
        corrected = (corrected / peak2 * 0.95).astype(np.float32)
    else:
        # Pipeline produced near-silent output — fall back to normalized original
        print("[Pipeline] WARNING: corrected signal near-zero, using original signal")
        corrected = signal.copy()
        peak3 = np.max(np.abs(corrected))
        if peak3 > 1e-8:
            corrected = (corrected / peak3 * 0.95).astype(np.float32)

    corrected_duration = len(corrected) / sr
    duration_removed   = max(0.0, original_duration - corrected_duration)
    disfluency_pct     = (duration_removed / original_duration * 100) if original_duration > 0 else 0.0

    return {
        "corrected_signal":    corrected,
        "original_duration":   original_duration,
        "corrected_duration":  corrected_duration,
        "disfluency_removed":  disfluency_pct,
        "prolongation_events": prolong_stats.get("prolongation_events", 0),
        "pause_events":        pause_stats.get("pauses_found", 0),
        "repetition_events":   rep_stats.get("repetition_events", 0),
        "sr":                  sr,
    }
