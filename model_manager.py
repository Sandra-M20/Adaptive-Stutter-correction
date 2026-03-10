"""
model_manager.py
================
Model Parameter Persistence Manager

Handles save/load operations for:
  - Reptile MAML meta-parameters (JSON)
  - Per-speaker adaptation history (JSON)
  - Training run metadata (JSON)

This module decouples I/O concerns from the adaptive_optimizer.py
algorithm logic, making the codebase easier to maintain and extend.
"""

import os
import json
import time
from config import MODEL_DIR, MAML_SAVE_PATH


class ModelManager:
    """
    Manages persistence of model parameters across sessions.

    Parameters
    ----------
    save_dir : str — directory for model files
    """

    def __init__(self, save_dir: str = MODEL_DIR):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #

    def save_maml_params(self, params: dict, history: list = None,
                         filename: str = "maml_params.json") -> str:
        """
        Persist Reptile MAML parameters and training history.

        Parameters
        ----------
        params   : dict  — current meta-parameters
        history  : list  — list of disfluency scores (optional)
        filename : str   — output JSON filename

        Returns
        -------
        path : str — absolute path to saved file
        """
        data = {
            "params":    params,
            "history":   (history or [])[-200:],  # keep last 200 entries
            "saved_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "version":   "1.0",
        }
        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[ModelManager] Saved MAML params -> {path}")
        return path

    def load_maml_params(self, filename: str = "maml_params.json") -> dict:
        """
        Load MAML parameters from JSON.

        Returns
        -------
        data : dict — {"params": ..., "history": ..., "saved_at": ...}
                      or empty dict if file not found.
        """
        path = os.path.join(self.save_dir, filename)
        if not os.path.exists(path):
            print(f"[ModelManager] No saved params at '{path}'. Using defaults.")
            return {}
        with open(path) as f:
            data = json.load(f)
        print(f"[ModelManager] Loaded MAML params from '{path}' "
              f"(saved at {data.get('saved_at', 'unknown')})")
        return data

    # ------------------------------------------------------------------ #

    def save_training_run(self, run_info: dict,
                          filename: str = "training_run.json") -> str:
        """
        Save metadata from a training run (accuracy, epoch, dataset, etc.)
        """
        run_info["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(run_info, f, indent=2)
        print(f"[ModelManager] Training run saved -> {path}")
        return path

    def load_training_run(self, filename: str = "training_run.json") -> dict:
        """Load most recent training run metadata."""
        path = os.path.join(self.save_dir, filename)
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------ #

    def list_saved_models(self) -> list:
        """Return a list of all JSON files in the model directory."""
        return [f for f in os.listdir(self.save_dir) if f.endswith(".json")]

    def checkpoint(self, optimizer, tag: str = "checkpoint") -> str:
        """
        Create a named checkpoint of an adaptive_optimizer's params.
        tag : str — label for the checkpoint (e.g. 'epoch_5', 'best')
        """
        fname = f"maml_{tag}.json"
        return self.save_maml_params(optimizer.params, optimizer.history, fname)
