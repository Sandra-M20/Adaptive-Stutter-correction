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
from config import TARGET_SR, FRAME_MS, HOP_MS, PAUSE_THRESHOLD_S, SIM_THRESHOLD, MIN_PROLONG_FRAMES


# ─────────────────────────────────────────────────────────────────────────────
# Overlap-Add reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _crossfade_join(chunk_a: np.ndarray, chunk_b: np.ndarray,
                    crossfade_ms: int = 20, sr: int = 22050) -> np.ndarray:
    """
    Join two audio chunks with a smooth cosine crossfade of crossfade_ms.
    Use this at splice points instead of hard concatenation.
    """
    fade_samples = int(sr * crossfade_ms / 1000)
    fade_samples = min(fade_samples, len(chunk_a), len(chunk_b))
    if fade_samples < 2:
        return np.concatenate([chunk_a, chunk_b])

    fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))
    fade_in  = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))

    result = np.concatenate([
        chunk_a[:-fade_samples],
        chunk_a[-fade_samples:] * fade_out + chunk_b[:fade_samples] * fade_in,
        chunk_b[fade_samples:]
    ])
    return result.astype(np.float32)


def _reconstruct(frames: list, hop_size: int, sr: int) -> np.ndarray:
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
    out = out.astype(np.float32)

    # Add explicit crossfade at splice points (detected by low frame similarity)
    if len(frames) < 2:
        return out

    def _frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    splice_frames = []
    for i in range(1, len(frames)):
        if _frame_similarity(frames[i - 1], frames[i]) < 0.6:
            splice_frames.append(i)

    if not splice_frames:
        return out

    shift = 0
    for i in splice_frames:
        splice_sample = i * hop_size - shift
        if splice_sample <= 0 or splice_sample >= len(out):
            continue
        left = out[:splice_sample]
        right = out[splice_sample:]
        joined = _crossfade_join(left, right, crossfade_ms=20, sr=sr)
        shift += (len(left) + len(right) - len(joined))
        out = joined

    return out


def _post_denoise(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply a gentle Wiener-style denoise pass to clean up splice
    artifacts introduced by OLA reconstruction.
    Uses a conservative over-subtraction of 0.8 to avoid speech damage.
    """
    try:
        try:
            from noise_reduction import NoiseReducer
        except Exception:
            from noise_reduction_professional import NoiseReducer
        reducer = NoiseReducer(over_subtraction_factor=0.8)
        denoised = reducer.reduce_noise(signal, sr)
        # Safety: if denoised RMS drops more than 40%, reject and return original
        rms_orig = np.sqrt(np.mean(signal ** 2))
        rms_new  = np.sqrt(np.mean(denoised ** 2))
        if rms_orig > 1e-8 and (rms_new / rms_orig) < 0.6:
            print("[Pipeline] Post-denoise rejected (too aggressive), using pre-denoise")
            return signal
        return denoised.astype(np.float32)
    except Exception as e:
        print(f"[Pipeline] Post-denoise failed ({e}), skipping")
        return signal


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
                 pause_threshold: float      = PAUSE_THRESHOLD_S,   # 500ms before flagging a block
                 retain_ratio: float         = 0.25,
                 similarity_threshold: float = SIM_THRESHOLD,   # extremely stable frames only
                 min_prolong_frames: int     = MIN_PROLONG_FRAMES) -> dict:  # 300ms minimum — real prolongations only
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
        hop_ms=HOP_MS,
    )
    frames, labels, prolong_stats = prolong_corrector.correct(frames, labels)

    # ── Step 6: Overlap-Add reconstruction ───────────────────────────────────
    reconstructed = _reconstruct(frames, hop_size, sr)

    if len(reconstructed) == 0:
        # Fallback: nothing to reconstruct — return original
        reconstructed = signal.copy()

    # ── Step 7: Repetition correction ────────────────────────────────────────
    rep_corrector = RepetitionCorrector(
        sr=sr,
        sim_threshold=0.75,           # more sensitive — catch real repetitions
        max_total_removal_ratio=0.30,
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

    # ── Step 9: Post-correction denoising (gentle) ──────────────────────
    corrected = _post_denoise(corrected, sr)

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
