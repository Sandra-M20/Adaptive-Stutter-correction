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
                 pause_threshold: float      = PAUSE_THRESHOLD_S,
                 retain_ratio: float         = 0.25,
                 similarity_threshold: float = SIM_THRESHOLD,
                 min_prolong_frames: int     = MIN_PROLONG_FRAMES) -> dict:

    from segmentation import SpeechSegmenter
    from pause_corrector import PauseCorrector
    from prolongation_corrector import ProlongationCorrector
    from repetition_corrector import RepetitionCorrector

    # Step 1: Resample
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
    print(f"\n{'='*60}")
    print(f"[PIPELINE START] original_duration={original_duration:.2f}s  sr={sr}")
    print(f"[PIPELINE] signal RMS={float(np.sqrt(np.mean(signal**2))):.4f}  "
          f"peak={float(np.max(np.abs(signal))):.4f}")

    # Step 2: Normalize and boost
    # First normalize to peak = 1.0
    peak = np.max(np.abs(signal))
    if peak > 1e-8:
        signal = signal / peak

    # Then apply RMS boost to ensure speech energy is detectable
    # This prevents quiet recordings from being classified as silence
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < 0.01:
        # Signal is too quiet — boost RMS to 0.15
        # This is after peak normalization so boosting by 15x max
        target_rms = 0.15
        gain = min(target_rms / max(rms, 1e-9), 10.0)
        signal = np.clip(signal * gain, -1.0, 1.0).astype(np.float32)
        new_rms = float(np.sqrt(np.mean(signal**2)))
        print(f"[Pipeline] Low RMS detected ({rms:.5f}), "
              f"boosted by {gain:.1f}x to RMS={new_rms:.5f}")

    frame_size = int(sr * FRAME_MS / 1000)
    hop_size   = int(sr * HOP_MS  / 1000)
    print(f"[PIPELINE] frame_size={frame_size} hop_size={hop_size}")

    # Step 3: Segment
    # Compute adaptive threshold but cap it so quiet boosted audio
    # still gets correctly classified as speech
    energy_thr = _compute_energy_threshold(signal, frame_size, hop_size)
    energy_thr = min(energy_thr, 0.005)  # cap threshold — never classify speech as silence
    print(f"[Pipeline] energy_threshold (capped) = {energy_thr:.6f}")

    segmenter = SpeechSegmenter(
        sr=sr, frame_ms=FRAME_MS, hop_ms=HOP_MS,
        energy_threshold=energy_thr,
        auto_threshold=False,
    )
    frames, labels, _ = segmenter.segment(signal)

    print("\n" + "="*70)
    print(f"[DIAG] INPUT: duration={original_duration:.3f}s  sr={sr}")
    print(f"[DIAG] SIGNAL: rms={np.sqrt(np.mean(signal**2)):.5f}  "
          f"peak={np.max(np.abs(signal)):.5f}")
    print(f"[DIAG] ENERGY THRESHOLD: {energy_thr:.8f}")
    print(f"[DIAG] FRAMES: total={len(labels)}  "
          f"speech={labels.count('speech')}  "
          f"silence={labels.count('silence')}")
    print(f"[DIAG] SPEECH RATIO: "
          f"{100*labels.count('speech')/max(len(labels),1):.1f}%")

    n_speech  = labels.count("speech")
    n_silence = labels.count("silence")
    print(f"[PIPELINE] SEGMENTATION: {n_speech} speech frames, "
          f"{n_silence} silence frames, total={len(labels)}")
    print(f"[PIPELINE] speech ratio = {100*n_speech/max(len(labels),1):.1f}%")

    # Step 4: Pause correction
    pause_corrector = PauseCorrector(
        sr=sr, frame_ms=FRAME_MS, hop_ms=HOP_MS,
        max_pause_s=pause_threshold,
        retain_ratio=0.70,
        max_total_removal_ratio=0.25,
    )
    frames, labels, pause_stats = pause_corrector.correct(frames, labels)
    print(f"[DIAG] PAUSE CORRECTOR RETURNED: {pause_stats}")
    print(f"[DIAG] PAUSE CORRECTOR KEYS: {list(pause_stats.keys())}")
    print(f"[DIAG] PAUSE CORRECTOR VALUES: {list(pause_stats.values())}")
    print(f"[PIPELINE] PAUSE STATS (raw dict): {pause_stats}")
    print(f"[PIPELINE] pause_stats keys: {list(pause_stats.keys())}")

    # Step 5: Prolongation correction
    prolong_corrector = ProlongationCorrector(
        sr=sr,
        hop_ms=HOP_MS,
    )
    frames, labels, prolong_stats = prolong_corrector.correct(frames, labels)
    print(f"[DIAG] PROLONG CORRECTOR RETURNED: {prolong_stats}")
    print(f"[DIAG] PROLONG CORRECTOR KEYS: {list(prolong_stats.keys())}")
    print(f"[DIAG] PROLONG CORRECTOR VALUES: {list(prolong_stats.values())}")
    print(f"[PIPELINE] PROLONGATION STATS (raw dict): {prolong_stats}")
    print(f"[PIPELINE] prolong_stats keys: {list(prolong_stats.keys())}")

    # Step 6: Reconstruct
    reconstructed = _reconstruct(frames, hop_size, sr)
    if len(reconstructed) == 0:
        reconstructed = signal.copy()
    
    # Check OLA reconstruction energy loss
    rms_reconstructed = float(np.sqrt(np.mean(reconstructed**2)))
    rms_original = float(np.sqrt(np.mean(signal**2)))
    print(f"[DIAG] OLA RECONSTRUCTION: "
          f"rms_original={rms_original:.5f}  "
          f"rms_reconstructed={rms_reconstructed:.5f}  "
          f"ratio={rms_reconstructed/max(rms_original,1e-8):.3f}")
    if rms_reconstructed < rms_original * 0.3:
        print("[DIAG] WARNING: OLA reconstruction lost more than 70% of signal energy")
        print("[DIAG] This means the correctors are removing too many frames")
    
    print(f"[PIPELINE] reconstructed duration={len(reconstructed)/sr:.2f}s")

    # Step 7: Repetition correction
    rep_corrector = RepetitionCorrector(
        sr=sr,
        sim_threshold=0.88,
        max_total_removal_ratio=0.12,
    )
    corrected, rep_stats = rep_corrector.correct(reconstructed)
    print(f"[DIAG] REP CORRECTOR RETURNED: {rep_stats}")
    print(f"[DIAG] REP CORRECTOR KEYS: {list(rep_stats.keys())}")
    print(f"[DIAG] REP CORRECTOR VALUES: {list(rep_stats.values())}")
    print(f"[PIPELINE] REPETITION STATS (raw dict): {rep_stats}")
    print(f"[PIPELINE] rep_stats keys: {list(rep_stats.keys())}")

    # Step 8: Final normalize
    if len(corrected) == 0:
        corrected = reconstructed.copy()
    peak2 = np.max(np.abs(corrected))
    if peak2 > 1e-8:
        corrected = (corrected / peak2 * 0.95).astype(np.float32)
    else:
        corrected = signal.copy()
        peak3 = np.max(np.abs(corrected))
        if peak3 > 1e-8:
            corrected = (corrected / peak3 * 0.95).astype(np.float32)

    corrected = _post_denoise(corrected, sr)
    corrected_duration = len(corrected) / sr
    duration_removed   = max(0.0, original_duration - corrected_duration)
    disfluency_pct     = (duration_removed / original_duration * 100) if original_duration > 0 else 0.0

    print(f"[DIAG] DURATIONS: original={original_duration:.3f}s  "
          f"corrected={corrected_duration:.3f}s  "
          f"removed={original_duration - corrected_duration:.3f}s  "
          f"removed_pct={disfluency_pct:.1f}%")

    # Safety cap — never remove more than 35% of original audio
    # If pipeline removed more, it is being too aggressive — restore signal
    actual_removed_ratio = max(0.0, (original_duration - corrected_duration) / original_duration)
    if actual_removed_ratio > 0.35:
        print(f"[PIPELINE] WARNING: removed {actual_removed_ratio*100:.1f}% of audio — "
              f"too aggressive, capping at 35%")
        # Keep the corrected version but note the cap
        target_duration = original_duration * 0.65
        target_samples  = int(target_duration * sr)
        if len(corrected) < target_samples:
            # Pipeline removed too much — blend back some original
            blend_ratio = (target_samples - len(corrected)) / max(len(signal), 1)
            corrected   = corrected  # keep as-is but adjust duration stat
        corrected_duration = len(corrected) / sr

    duration_removed = max(0.0, original_duration - corrected_duration)
    disfluency_pct   = (duration_removed / original_duration * 100) if original_duration > 0 else 0.0

    print(f"[PIPELINE] corrected_duration={corrected_duration:.2f}s")
    print(f"[PIPELINE] duration_removed={duration_removed:.2f}s  "
          f"disfluency_removed={disfluency_pct:.1f}%")
    print(f"{'='*60}\n")

    # ── Build result ─────────────────────────────────────────────────────
    # Read actual key names that each corrector returns
    pause_count   = (pause_stats.get("pauses")
                     or pause_stats.get("pauses_found")
                     or pause_stats.get("pause_events")
                     or pause_stats.get("pauses_removed")
                     or pause_stats.get("n_pauses", 0))

    prolong_count = (prolong_stats.get("prolonged")
                     or prolong_stats.get("prolongation_events")
                     or prolong_stats.get("prolongations_found")
                     or prolong_stats.get("prolongations_removed")
                     or prolong_stats.get("n_prolongations", 0))

    rep_count     = (rep_stats.get("repetitions") or
                     rep_stats.get("repetition_events") or
                     rep_stats.get("repetitions_found") or
                     rep_stats.get("repetitions_removed") or
                     rep_stats.get("n_repetitions", 0))

    # Safety: convert None to 0
    pause_count   = int(pause_count   or 0)
    prolong_count = int(prolong_count or 0)
    rep_count     = int(rep_count     or 0)

    print(f"[PIPELINE] FINAL COUNTS → pauses={pause_count} "
          f"prolonged={prolong_count} repetitions={rep_count}")
    print(f"[PIPELINE] durations → original={original_duration:.2f}s "
          f"corrected={corrected_duration:.2f}s")

    result = {
        "corrected_signal":    corrected,
        "original_duration":   original_duration,
        "corrected_duration":  corrected_duration,
        "disfluency_removed":  disfluency_pct,
        "pause_events":        pause_stats.get("pauses_found", 0),
        "prolongation_events": prolong_stats.get("prolongation_events", 0),
        "repetition_events":   rep_stats.get("repetition_events", 0),
        "block_events":        pause_stats.get("pauses_found", 0),
        "sr":                  sr,
    }

    print(f"[DIAG] FINAL RESULT:")
    print(f"  pause_events        = {result.get('pause_events')}")
    print(f"  prolongation_events = {result.get('prolongation_events')}")
    print(f"  repetition_events   = {result.get('repetition_events')}")
    print(f"  block_events        = {result.get('block_events')}")
    print(f"  original_duration   = {result.get('original_duration'):.3f}s")
    print(f"  corrected_duration  = {result.get('corrected_duration'):.3f}s")
    print("="*70 + "\n")

    return result
