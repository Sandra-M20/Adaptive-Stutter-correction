"""
train.py
========
Training Script: Reptile MAML on UCLASS Archive

Trains the Reptile meta-learning optimizer across multiple audio clips
from the archive dataset and saves the adapted meta-parameters.

Training loop:
  For each epoch:
    For each mini-batch of K dysfluent clips:
      1. Load audio clip
      2. Preprocess (denoise, resample, normalize)
      3. Run inner-loop Reptile adaptation
      4. Outer Reptile update meta-params
  Save checkpoint per epoch.

This produces a trained `model/maml_params.json` file that the
production pipeline can load to start each session with better
default thresholds instead of cold-starting from scratch.

Usage:
  python train.py --epochs 3 --batch_size 5 --max_clips 30
"""

import argparse
import os
import time
import numpy as np
import soundfile as sf

from config import TARGET_SR, MAML_SAVE_PATH, MODEL_DIR
from adaptive_optimizer import ReptileMAML
from model_manager import ModelManager
from dataset_loader import DatasetLoader


def train(epochs: int = 3, batch_size: int = 5,
          max_clips: int = 30, seed: int = 42):
    """
    Main training function.

    Parameters
    ----------
    epochs     : int — number of full passes over the dataset
    batch_size : int — clips per Reptile meta-update step
    max_clips  : int — maximum clips to use from the archive
    seed       : int — random seed for reproducibility
    """
    print("=" * 60)
    print("  Reptile MAML Training on UCLASS Stuttered Speech Dataset")
    print("=" * 60)

    # 1. Load dataset
    loader   = DatasetLoader()
    samples  = loader.sample_batch(max_clips, dysfluent_only=True, seed=seed)
    loader.stats()
    print(f"Training on {len(samples)} dysfluent clips | {epochs} epochs | "
          f"batch_size={batch_size}\n")

    # 2. Initialize
    optimizer = ReptileMAML()
    manager   = ModelManager(save_dir=MODEL_DIR)

    # Try loading existing checkpoint to continue from prior training
    existing = manager.load_maml_params()
    if existing.get("params"):
        optimizer.params = existing["params"]
        optimizer.history = existing.get("history", [])
        print(f"Resumed from existing checkpoint: {optimizer.params}\n")

    # 3. Training loop
    all_scores = []
    t_start    = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        epoch_scores = []

        # Shuffle samples
        np.random.seed(seed + epoch)
        indices = np.random.permutation(len(samples))

        for batch_start in range(0, len(samples), batch_size):
            batch_idx = indices[batch_start: batch_start + batch_size]
            print(f"  Batch {batch_start//batch_size + 1}: "
                  f"{len(batch_idx)} clips")

            for idx in batch_idx:
                s = samples[int(idx)]
                try:
                    signal, sr = _load_and_preprocess(s["path"])
                    params = optimizer.adapt(signal, sr)
                    epoch_scores.append(optimizer.history[-1] if optimizer.history else 0)
                except Exception as e:
                    print(f"    SKIP {s['path']}: {e}")

            # Checkpoint after each batch
            manager.checkpoint(optimizer, tag=f"epoch{epoch}_batch{batch_start}")

        mean_score = np.mean(epoch_scores) if epoch_scores else 0
        all_scores.extend(epoch_scores)
        print(f"  Epoch {epoch} done. Mean disfluency score: {mean_score:.4f}")
        print(f"  Meta-params: {optimizer.params}")

    # 4. Final save
    elapsed = time.time() - t_start
    run_info = {
        "epochs":          epochs,
        "batch_size":      batch_size,
        "clips_used":      len(samples),
        "final_params":    optimizer.params,
        "final_mean_score": float(np.mean(all_scores)) if all_scores else None,
        "training_time_s": round(elapsed, 2),
    }
    manager.save_maml_params(optimizer.params, optimizer.history)
    manager.save_training_run(run_info)

    print("\n" + "=" * 60)
    print("  Training Complete")
    print(f"  Time elapsed    : {elapsed:.1f}s")
    print(f"  Final params    : {optimizer.params}")
    print(f"  MAML params saved -> {MAML_SAVE_PATH}")
    print("=" * 60)
    return run_info


def _load_and_preprocess(path: str):
    """Load audio file and run Step 1-2 preprocessing."""
    from preprocessing import AudioPreprocessor
    proc = AudioPreprocessor(target_sr=TARGET_SR, noise_reduce=True)
    signal, sr = proc.process(path)
    return signal, sr


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Reptile MAML on UCLASS stuttered speech dataset")
    parser.add_argument("--epochs",     type=int, default=3,  help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=5,  help="Clips per batch")
    parser.add_argument("--max_clips",  type=int, default=30, help="Max dataset clips")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(epochs=args.epochs,
          batch_size=args.batch_size,
          max_clips=args.max_clips,
          seed=args.seed)
