"""
metrics.py
==========
Individual Evaluation Metric Functions

Provides standalone metric functions that can be imported by evaluator.py
and wer_evaluator.py. Designed to be reusable across all evaluation contexts.

Metrics:
  - wer()                  : Word Error Rate
  - levenshtein()          : Edit distance between two word lists
  - duration_reduction()   : How much the audio was shortened (%)
  - fluency_ratio()        : Percentage of speech frames (vs. total)
  - prolongation_rate()    : Average cosine similarity in speech frames
  - disfluency_score()     : Composite score (lower = more fluent)
  - snr()                  : Signal-to-Noise Ratio estimate (dB)
"""

import math
import numpy as np
from utils import cosine_similarity, short_time_energy
from config import TARGET_SR, FRAME_MS, ENERGY_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# WORD ERROR RATE
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein(ref: list, hyp: list) -> int:
    """
    Compute Levenshtein (edit) distance between two token lists.
    Operations: insertion, deletion, substitution (each cost 1).
    """
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        new_dp = [i] + [0] * n
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j],       # deletion
                                    new_dp[j - 1], # insertion
                                    dp[j - 1])     # substitution
        dp = new_dp
    return dp[n]


def wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate = Levenshtein(ref, hyp) / len(ref).
    Tokens are lowercased and stripped of punctuation before comparison.
    Returns a float in [0, inf) (can exceed 1.0 for long insertions).
    """
    import re
    clean = lambda s: re.sub(r"[^a-z0-9\s]", "", s.lower()).split()
    ref = clean(reference)
    hyp = clean(hypothesis)
    if not ref:
        return 0.0 if not hyp else float("inf")
    return levenshtein(ref, hyp) / len(ref)


# ─────────────────────────────────────────────────────────────────────────────
# DURATION & FLUENCY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def duration_reduction(original: np.ndarray, corrected: np.ndarray) -> float:
    """
    Percentage reduction in audio length after correction.
    Positive = shorter (good), Negative = longer (rare edge case).
    """
    if len(original) == 0:
        return 0.0
    return (1.0 - len(corrected) / len(original)) * 100.0


def fluency_ratio(signal: np.ndarray, sr: int = TARGET_SR,
                  frame_ms: int = FRAME_MS,
                  threshold: float = ENERGY_THRESHOLD) -> float:
    """
    Ratio of speech frames to total frames (as a percentage).
    Higher = more fluent / less silence.
    """
    frame_size = int(sr * frame_ms / 1000)
    energies   = [short_time_energy(signal[s:s + frame_size])
                  for s in range(0, len(signal) - frame_size + 1, frame_size)]
    if not energies:
        return 0.0
    n_speech = sum(1 for e in energies if e > threshold)
    return n_speech / len(energies) * 100.0


def prolongation_rate(signal: np.ndarray, sr: int = TARGET_SR,
                      frame_ms: int = FRAME_MS) -> float:
    """
    Estimate prolongation intensity as the mean cosine similarity
    between adjacent speech frames. Higher = more prolonged speech.
    """
    from feature_extractor import FeatureExtractor
    fe         = FeatureExtractor(sr=sr)
    frame_size = int(sr * frame_ms / 1000)
    frames     = [signal[s:s + frame_size]
                  for s in range(0, len(signal) - frame_size + 1, frame_size)]
    if len(frames) < 2:
        return 0.0
    feats = [fe.extract(f) for f in frames]
    sims  = [cosine_similarity(feats[i], feats[i - 1]) for i in range(1, len(feats))]
    return float(np.mean(sims)) if sims else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Estimate Signal-to-Noise Ratio in decibels.
    SNR (dB) = 10 * log10( signal_power / noise_power )
    The noise is approximated as: noisy - clean.
    """
    min_len = min(len(clean), len(noisy))
    if min_len == 0:
        return 0.0
    c = clean[:min_len]
    n = noisy[:min_len]
    signal_pwr = np.mean(c ** 2)
    noise_pwr  = np.mean((n - c) ** 2)
    if noise_pwr < 1e-12:
        return float("inf")
    if signal_pwr < 1e-12:
        return -float("inf")
    return 10.0 * math.log10(signal_pwr / noise_pwr)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE DISFLUENCY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def disfluency_score(signal: np.ndarray, sr: int = TARGET_SR) -> float:
    """
    Composite disfluency score in [0, 1].
    Combines fluency_ratio and prolongation_rate.
    Lower score = more fluent / better corrected.
    """
    flu  = fluency_ratio(signal, sr)          # % of speech frames
    prol = prolongation_rate(signal, sr)      # mean inter-frame similarity
    # Higher fluency ratio = more speech = potentially more stuttering retained
    # Higher prolongation rate = more prolonged sounds remain
    # Scale both to [0,1] weights
    flu_term  = (100.0 - flu) / 100.0        # silence fraction
    prol_term = max(0.0, (prol - 0.5) * 2)  # excess similarity above 0.5
    return float(0.5 * flu_term + 0.5 * prol_term)
