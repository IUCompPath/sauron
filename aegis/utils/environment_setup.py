# aegis/utils/environment_setup.py
import argparse
import os
import random
from typing import Optional

import numpy as np
import torch


def setup_device() -> torch.device:
    """Sets up and returns the device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device is set to: {device}")
    return device


def seed_everything(seed: int):
    """Seeds random number generators for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        # The following two lines are often recommended for reproducibility with CuDNN
        # However, they can impact performance. Use with caution.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Seeded everything with seed: {seed}")


def create_results_directory(
    base_results_dir: str, exp_code: str, seed: int, fold_num: Optional[int] = None
) -> str:
    """
    Creates the results directory for the experiment.
    If fold_num is provided, creates a subdirectory for that fold.
    """
    experiment_path = os.path.join(base_results_dir, f"{exp_code}_s{seed}")
    if fold_num is not None:
        results_path = os.path.join(experiment_path, f"fold_{fold_num}")
    else:
        results_path = experiment_path  # Main experiment directory

    os.makedirs(results_path, exist_ok=True)
    return results_path


def log_experiment_details(args: argparse.Namespace, results_dir: str):
    """Logs experiment arguments to a file."""
    # Ensure results_dir here is the main experiment directory, not a fold-specific one
    # if log_experiment_details is called once per experiment.
    # If called per fold, then it's fine if results_dir is fold-specific.
    # For now, assuming it's called once for the main experiment.

    # If results_dir might be a fold-specific path, get the parent experiment path
    # e.g., experiment_base_path = os.path.dirname(results_dir) if "fold_" in os.path.basename(results_dir) else results_dir
    experiment_base_path = (
        results_dir  # Assuming results_dir passed is already the main exp dir
    )

    filepath = os.path.join(experiment_base_path, "experiment_args.txt")
    with open(filepath, "w") as f:
        for key, val in sorted(vars(args).items()):  # Sort for consistent order
            f.write(f"{key}: {val}\n")
    print(f"Experiment arguments logged to: {filepath}")
