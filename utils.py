"""
utils.py
========
Shared pure-numpy DSP utility functions used across all pipeline modules.

Contains:
  - _resample()              : Linear interpolation resampler
  - _mel_filterbank()        : Triangular mel filterbank matrix
  - _stft()                  : Short-Time Fourier Transform
  - _istft()                 : Inverse STFT (overlap-add)
  - _compute_mfcc()          : Mel Frequency Cepstral Coefficients
  - _compute_lpc()           : Linear Predictive Coding (Levinson-Durbin)
  - _cosine_similarity()     : Cosine similarity between two vectors
  - _dtw_distance()          : Dynamic Time Warping distance
  - normalize()              : Peak normalize a signal to [-1, 1]
"""

import math
import numpy as np
from config import (N_FFT, HOP_STFT, N_MFCC, N_MEL_FILTERS,
                    LPC_ORDER, TARGET_SR)


# -----------------------------------------------------------------------------
# RESAMPLING
# -----------------------------------------------------------------------------

def resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample `signal` from `orig_sr` to `target_sr` using linear interpolation.
    Avoids scipy / librosa to prevent DLL crashes on Windows.
    """
    if orig_sr == target_sr:
        return signal.astype(np.float32)
    n_out = int(len(signal) * target_sr / orig_sr)
    old_indices = np.linspace(0, len(signal) - 1, n_out)
    return np.interp(old_indices, np.arange(len(signal)), signal).astype(np.float32)


# -----------------------------------------------------------------------------
# NORMALIZATION
# -----------------------------------------------------------------------------

def normalize(signal: np.ndarray) -> np.ndarray:
    """Peak-normalize signal amplitude to the range [-1.0, 1.0]."""
    peak = np.max(np.abs(signal))
    return (signal / peak).astype(np.float32) if peak > 1e-8 else signal


# -----------------------------------------------------------------------------
# MEL FILTERBANK
# -----------------------------------------------------------------------------

def mel_filterbank(n_filters: int, n_fft: int, sr: int,
                   fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """
    Build triangular mel filterbank matrix of shape (n_filters, n_fft//2+1).
    Used for MFCC computation and Whisper mel spectrogram generation.
    """
    if fmax is None:
        fmax = sr / 2.0
    hz2mel = lambda f: 2595.0 * math.log10(1 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1)

    mel_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_filters + 2)
    hz_pts  = mel2hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    banks = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        if c > lo:
            banks[m - 1, lo:c] = (np.arange(lo, c) - lo) / (c - lo)
        if hi > c:
            banks[m - 1, c:hi] = (hi - np.arange(c, hi)) / (hi - c)
    return banks


# -----------------------------------------------------------------------------
# STFT / iSTFT
# -----------------------------------------------------------------------------

def stft(signal: np.ndarray, n_fft: int = N_FFT,
         hop: int = HOP_STFT) -> np.ndarray:
    """
    Compute complex Short-Time Fourier Transform.
    Returns complex matrix of shape (T_frames, n_fft//2+1).
    """
    window = np.hanning(n_fft)
    frames = []
    for s in range(0, len(signal) - n_fft + 1, hop):
        frames.append(np.fft.rfft(signal[s:s + n_fft] * window, n=n_fft))
    return np.array(frames)    # (T, F)


def istft(stft_frames: np.ndarray, hop: int = HOP_STFT,
          signal_len: int = None) -> np.ndarray:
    """
    Inverse STFT via Overlap-Add.
    Returns reconstructed signal of length `signal_len` (or inferred).
    """
    n_fft   = (stft_frames.shape[1] - 1) * 2
    window  = np.hanning(n_fft)
    out_len = signal_len or ((len(stft_frames) - 1) * hop + n_fft)
    out     = np.zeros(out_len, dtype=np.float32)
    wsum    = np.zeros(out_len, dtype=np.float32)
    for i, frame in enumerate(stft_frames):
        s, e = i * hop, i * hop + n_fft
        if e > out_len:
            break
        out[s:e]  += np.real(np.fft.irfft(frame, n=n_fft)) * window
        wsum[s:e] += window ** 2
    out[wsum > 1e-8] /= wsum[wsum > 1e-8]
    return out


# -----------------------------------------------------------------------------
# MFCC
# -----------------------------------------------------------------------------

# Global cache to avoid redundant filterbank generation
_MFCC_FILTERBANK_CACHE = {}

def compute_mfcc(frame: np.ndarray, sr: int = TARGET_SR,
                 n_mfcc: int = N_MFCC,
                 n_filters: int = N_MEL_FILTERS,
                 n_fft: int = N_FFT) -> np.ndarray:
    """
    Compute `n_mfcc` MFCC coefficients for a single audio frame.
    Vectorized version for performance.
    """
    # 1. Framing and Windowing
    windowed = frame * np.hanning(len(frame))
    
    # 2. Power Spectrum
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
    
    # 3. Mel Filterbank (Cached)
    cache_key = (n_filters, n_fft, sr)
    if cache_key not in _MFCC_FILTERBANK_CACHE:
        _MFCC_FILTERBANK_CACHE[cache_key] = mel_filterbank(n_filters, n_fft, sr)
    banks = _MFCC_FILTERBANK_CACHE[cache_key]
    
    # 4. Mel Energies (Log Scale)
    mel_e = np.log(np.maximum(banks @ spectrum[:n_fft // 2 + 1], 1e-10))
    
    # 5. Discrete Cosine Transform (DCT-II) - Vectorized
    # Pre-compute DCT basis if possible, or just use vectorized dot product
    N = n_filters
    k = np.arange(n_mfcc).reshape(-1, 1)
    n = np.arange(N)
    basis = np.cos(math.pi * k * (2 * n + 1) / (2 * N))
    mfcc = basis @ mel_e
    
    return mfcc.astype(np.float32)


# -----------------------------------------------------------------------------
# LPC  (Levinson-Durbin recursion)
# -----------------------------------------------------------------------------

def compute_lpc(frame: np.ndarray, order: int = LPC_ORDER) -> np.ndarray:
    """
    Compute `order` Linear Predictive Coding coefficients using the
    Levinson-Durbin recursion algorithm (pure numpy).
    """
    n = len(frame)
    r = np.array([np.sum(frame[:n - k] * frame[k:]) for k in range(order + 1)])
    if r[0] < 1e-10:
        return np.zeros(order, dtype=np.float32)
    a, E = np.zeros(order), r[0]
    for i in range(order):
        lam      = -(r[i + 1] + a[:i] @ r[i:0:-1]) / E
        a_new    = np.zeros(order)
        a_new[i] = lam
        for j in range(i):
            a_new[j] = a[j] + lam * a[i - 1 - j]
        a, E = a_new, E * (1.0 - lam ** 2)
    return a.astype(np.float32)


# -----------------------------------------------------------------------------
# SIMILARITY METRICS
# -----------------------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns value in [-1, 1]."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return 0.0 if n1 < 1e-10 or n2 < 1e-10 else float(np.dot(v1, v2) / (n1 * n2))


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Compute normalized Dynamic Time Warping distance between two
    feature sequences seq1 (T1, D) and seq2 (T2, D).
    Lower = more similar.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float("inf")
    cost = np.full((n + 1, m + 1), float("inf"))
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = float(np.linalg.norm(seq1[i - 1] - seq2[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m] / (n + m)


def short_time_energy(frame: np.ndarray) -> float:
    """Compute Short-Time Energy (mean squared amplitude) for a single frame."""
    return float(np.mean(frame ** 2))


def spectral_flux(frame1: np.ndarray, frame2: np.ndarray, n_fft: int = N_FFT) -> float:
    """
    Compute Spectral Flux between two frames.
    Formula: sum( (S_k(t) - S_k(t-1))^2 )
    """
    s1 = np.abs(np.fft.rfft(frame1 * np.hanning(len(frame1)), n=n_fft)) ** 2
    s2 = np.abs(np.fft.rfft(frame2 * np.hanning(len(frame2)), n=n_fft)) ** 2
    
    # Normalize spectra to make flux independent of absolute volume
    s1 /= (np.sum(s1) + 1e-10)
    s2 /= (np.sum(s2) + 1e-10)
    
    return float(np.sum((s2 - s1) ** 2))


def spectral_flatness(frame: np.ndarray, n_fft: int = N_FFT) -> float:
    """
    Compute Spectral Flatness of a frame.
    Formula: Geometric Mean / Arithmetic Mean of the power spectrum.
    Values close to 1.0 = noise-like; closer to 0.0 = tonal (speech).
    """
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n=n_fft)) ** 2
    spec = np.maximum(spec, 1e-10)  # Avoid log(0)
    
    gmean = np.exp(np.mean(np.log(spec)))
    amean = np.mean(spec)
    
    return float(gmean / amean) if amean > 1e-10 else 0.0
