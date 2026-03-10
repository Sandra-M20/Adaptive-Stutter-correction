"""
visualizer.py
=============
Audio Visualization and Analysis Plots

Generates plots for the DSP pipeline stages:
  - plot_waveform()         : Raw audio waveform
  - plot_energy()           : Frame-level Short-Time Energy
  - plot_similarity()       : Inter-frame cosine similarity profile
  - plot_spectrogram()      : Time-frequency spectrogram (STFT magnitude)
  - plot_mfcc_heatmap()     : MFCC feature heatmap over time
  - plot_before_after()     : Side-by-side waveform comparison
  - plot_pipeline_summary() : All key metrics in one grid

Uses only matplotlib (no librosa). Saves to RESULTS_DIR.
"""

import os
import numpy as np
from config import TARGET_SR, FRAME_MS, ENERGY_THRESHOLD, RESULTS_DIR
from utils import short_time_energy, stft

_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")   # Non-interactive backend (no display required)
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("[Visualizer] WARNING: matplotlib not installed. "
          "Plots will be skipped. Run: pip install matplotlib")

def _no_plot(name):
    print(f"[Visualizer] Skipped '{name}' — matplotlib not available.")
    return None


def _can_plot(name):
    if _MATPLOTLIB_AVAILABLE:
        return True
    _no_plot(name)
    return False


os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _time_axis(signal: np.ndarray, sr: int) -> np.ndarray:
    return np.linspace(0, len(signal) / sr, len(signal))


def _energy_profile(signal: np.ndarray, sr: int,
                    frame_ms: int = FRAME_MS) -> tuple:
    frame_size = int(sr * frame_ms / 1000)
    energies   = [short_time_energy(signal[s:s + frame_size])
                  for s in range(0, len(signal) - frame_size + 1, frame_size)]
    times      = [(i + 0.5) * frame_ms / 1000 for i in range(len(energies))]
    return np.array(times), np.array(energies)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform(signal: np.ndarray, sr: int = TARGET_SR,
                 title: str = "Waveform", filename: str = None) -> str:
    """Save waveform plot as PNG. Returns file path."""
    if not _can_plot("plot_waveform"):
        return ""
    fig, ax = plt.subplots(figsize=(10, 3))
    t = _time_axis(signal, sr)
    ax.plot(t, signal, linewidth=0.4, color="#4a90d9")
    ax.set_xlim([0, t[-1]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename or f"{title.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Visualizer] Saved waveform -> {path}")
    return path


def plot_energy(signal: np.ndarray, sr: int = TARGET_SR,
                threshold: float = ENERGY_THRESHOLD,
                title: str = "STE Energy", filename: str = None) -> str:
    """Save STE energy profile with threshold line."""
    if not _can_plot("plot_energy"):
        return ""
    times, energies = _energy_profile(signal, sr)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(times, energies, alpha=0.6, color="#5cb85c", label="STE")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"Threshold={threshold}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename or "ste_energy.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Visualizer] Saved energy plot -> {path}")
    return path


def plot_similarity(similarities: list, threshold: float = 0.96,
                    title: str = "Inter-Frame Similarity",
                    filename: str = None) -> str:
    """Plot the cosine similarity between adjacent frames."""
    if not _can_plot("plot_similarity"):
        return ""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(similarities, color="#d9534f", linewidth=0.8, label="Cosine Similarity")
    ax.axhline(threshold, color="orange", linestyle="--",
               linewidth=1.2, label=f"Threshold={threshold}")
    ax.fill_between(range(len(similarities)),
                    [threshold] * len(similarities), similarities,
                    where=[s >= threshold for s in similarities],
                    alpha=0.3, color="orange", label="Prolongation Zone")
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename or "frame_similarity.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Visualizer] Saved similarity plot -> {path}")
    return path


def plot_spectrogram(signal: np.ndarray, sr: int = TARGET_SR,
                     title: str = "Spectrogram", filename: str = None) -> str:
    """Plot STFT magnitude spectrogram."""
    if not _can_plot("plot_spectrogram"):
        return ""
    frames = stft(signal)
    mag_db = 20 * np.log10(np.maximum(np.abs(frames.T), 1e-10))
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mag_db, origin="lower", aspect="auto",
                   cmap="inferno", vmin=mag_db.max() - 60)
    plt.colorbar(im, ax=ax, label="dB")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frequency Bin")
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename or "spectrogram.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Visualizer] Saved spectrogram -> {path}")
    return path


def plot_before_after(original: np.ndarray, corrected: np.ndarray,
                      sr: int = TARGET_SR, filename: str = None) -> str:
    """Side-by-side waveform comparison: original vs. corrected."""
    if not _can_plot("plot_before_after"):
        return ""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
    dur_orig = len(original) / sr
    dur_corr = len(corrected) / sr

    axes[0].plot(_time_axis(original, sr), original, "#d9534f", linewidth=0.4)
    axes[0].set_title(f"Stuttered Input  ({dur_orig:.2f}s)", fontsize=11)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(_time_axis(corrected, sr), corrected, "#5cb85c", linewidth=0.4)
    axes[1].set_title(f"Corrected Output ({dur_corr:.2f}s) "
                      f"[{100*(1-dur_corr/dur_orig):.1f}% shorter]", fontsize=11)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Stutter Correction: Before vs After", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename or "before_after.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[Visualizer] Saved before/after -> {path}")
    return path


def plot_pipeline_summary(original: np.ndarray, corrected: np.ndarray,
                          similarities: list, sr: int = TARGET_SR,
                          threshold: float = 0.96) -> str:
    """
    Combined 3-panel summary figure:
      Row 1: Before waveform
      Row 2: After waveform
      Row 3: Frame similarity with prolongation zones highlighted
    """
    if not _can_plot("plot_pipeline_summary"):
        return ""
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)
    cmap = {"before": "#d9534f", "after": "#5cb85c", "sim": "#9b59b6"}

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(_time_axis(original, sr), original, cmap["before"], lw=0.4)
    ax1.set_title(f"[Step 1-2] Stuttered Input ({len(original)/sr:.2f}s)", fontsize=10)
    ax1.grid(True, alpha=0.2); ax1.set_ylabel("Amplitude")

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(_time_axis(corrected, sr), corrected, cmap["after"], lw=0.4)
    ax2.set_title(f"[Step 11] Corrected Output ({len(corrected)/sr:.2f}s)", fontsize=10)
    ax2.grid(True, alpha=0.2); ax2.set_ylabel("Amplitude")

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(similarities, cmap["sim"], lw=0.8, label="Cosine Similarity")
    ax3.axhline(threshold, color="orange", linestyle="--", lw=1.2, label=f"Threshold {threshold}")
    ax3.fill_between(range(len(similarities)),
                     [threshold] * len(similarities), similarities,
                     where=[s >= threshold for s in similarities],
                     alpha=0.3, color="orange", label="Prolongation")
    ax3.set_title("[Steps 7-8] Inter-Frame Similarity (Prolongation Detection)", fontsize=10)
    ax3.set_ylim([-0.05, 1.05]); ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.2); ax3.set_xlabel("Frame Index")

    plt.suptitle("DSP Pipeline Analysis Summary", fontsize=14, fontweight="bold")
    path = os.path.join(RESULTS_DIR, "pipeline_summary.png")
    plt.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[Visualizer] Pipeline summary saved -> {path}")
    return path


def plot_maml_iterations(trace: list, filename_prefix: str = "maml") -> dict:
    """
    Generate report-style MAML iteration figures from optimizer trace.
    Returns dict of generated file paths.
    """
    if not _can_plot("plot_maml_iterations"):
        return {}
    if not trace:
        print("[Visualizer] No MAML trace available; skipping iteration plots.")
        return {}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    steps = [t.get("step", i + 1) for i, t in enumerate(trace)]
    loss = [float(t.get("loss", 1.0)) for t in trace]
    score = [float(t.get("score", 0.0)) for t in trace]
    e_thr = [float(t.get("params", {}).get("energy_threshold", np.nan)) for t in trace]
    s_thr = [float(t.get("params", {}).get("sim_threshold", np.nan)) for t in trace]
    p_thr = [float(t.get("params", {}).get("max_pause_s", np.nan)) for t in trace]

    out = {}

    # 7.1 - 7.10: per-iteration snapshots
    for idx, t in enumerate(trace[:10], start=1):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        params = t.get("params", {})
        txt = (
            f"Iteration {idx}\n"
            f"Loss: {t.get('loss', 1.0):.4f}\n"
            f"Score: {t.get('score', 0.0):.4f}\n"
            f"Energy Thr: {params.get('energy_threshold', 0.0):.5f}\n"
            f"Pause Thr(s): {params.get('max_pause_s', 0.0):.3f}\n"
            f"Similarity Thr: {params.get('sim_threshold', 0.0):.4f}"
        )
        ax.text(0.05, 0.95, txt, va="top", ha="left", family="monospace", fontsize=10)
        ax.set_axis_off()
        path = os.path.join(RESULTS_DIR, f"{filename_prefix}_iter_{idx:02d}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close(fig)
        out[f"iter_{idx}"] = path

    # 7.11: updated parameters trend
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(steps, e_thr, label="energy_threshold", marker="o")
    ax.plot(steps, p_thr, label="max_pause_s", marker="o")
    ax.plot(steps, s_thr, label="sim_threshold", marker="o")
    ax.set_title("Updated Parameters Across Iterations")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(RESULTS_DIR, f"{filename_prefix}_updated_parameters.png")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    out["updated_parameters"] = path

    # 7.12/7.13/7.14 threshold plots
    for name, vals, title, fname in [
        ("noise_presence_threshold", e_thr, "Noise Presence Threshold (Energy)", f"{filename_prefix}_noise_threshold.png"),
        ("correlation_threshold", s_thr, "Correlation Threshold (Similarity)", f"{filename_prefix}_correlation_threshold.png"),
        ("prolongation_threshold", score, "Prolongation Optimization Score", f"{filename_prefix}_prolongation_threshold.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(steps, vals, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        p = os.path.join(RESULTS_DIR, fname)
        plt.tight_layout()
        plt.savefig(p, dpi=130)
        plt.close(fig)
        out[name] = p

    print(f"[Visualizer] Saved MAML iteration figures ({len(out)} files).")
    return out
