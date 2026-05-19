import os
import platform
import sys
from unittest.mock import MagicMock

# Conditional import mocking for Mamba on Windows
if platform.system() == "Windows":
    # Mock mamba_ssm and its submodules to prevent ImportErrors/DLL errors on Windows
    # This allows the code to run even if mamba_ssm is not installed or compatible
    mamba_mock = MagicMock()
    sys.modules["mamba_ssm"] = mamba_mock
    sys.modules["mamba_ssm.modules"] = mamba_mock
    sys.modules["mamba_ssm.modules.bimamba"] = mamba_mock
    sys.modules["mamba_ssm.modules.mamba_simple"] = mamba_mock
    sys.modules["mamba_ssm.modules.srmamba"] = mamba_mock

    # Also mock the local aegis.mil_models.mamba_ssm path
    sys.modules["aegis.mil_models.mamba_ssm"] = mamba_mock
    sys.modules["aegis.mil_models.mamba_ssm.modules"] = mamba_mock
    sys.modules["aegis.mil_models.mamba_ssm.modules.bimamba"] = mamba_mock
    sys.modules["aegis.mil_models.mamba_ssm.modules.mamba_simple"] = mamba_mock
    sys.modules["aegis.mil_models.mamba_ssm.modules.srmamba"] = mamba_mock

from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from aegis.data.data_utils import get_dataloader
from aegis.training.lightning_module import aegis


def train_fold(
    train_dataset,
    val_dataset,
    test_dataset,
    cur_fold_num: int,
    args,
    experiment_base_results_dir: str,  # This should be the fold-specific dir
) -> Tuple:
    """
    Trains and evaluates a model for a single fold.
    """
    print(f"Initializing training for fold {cur_fold_num}...")

    # DataLoaders
    n_subsamples = getattr(args, "n_subsamples", None)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn_type": args.task_type,
        "n_subsamples": n_subsamples,
        "num_workers": getattr(
            args, "num_workers", 8
        ),  # Try reducing from 16 to 4 or 8 first
        "pin_memory": True,  # <--- ADD THIS
        "persistent_workers": True,  # <--- ADD THIS (Only works if num_workers > 0)
        "prefetch_factor": 4,  # Reduce from default 16 to prevent RAM spikes
    }

    train_loader = get_dataloader(
        train_dataset,
        shuffle=True,
        use_weighted_sampler=args.weighted_sample,
        **loader_kwargs,
    )

    # For validation and testing, we want to use all samples (n_subsamples=-1)
    # We remove n_subsamples from kwargs so get_dataloader uses the dataset's value
    eval_loader_kwargs = loader_kwargs.copy()
    eval_loader_kwargs["n_subsamples"] = None

    val_loader = get_dataloader(val_dataset, shuffle=False, **eval_loader_kwargs)
    test_loader = get_dataloader(test_dataset, shuffle=False, **eval_loader_kwargs)

    # Model
    model = aegis(args)

    # Callbacks
    checkpoint_dir = os.path.join(experiment_base_results_dir, "checkpoints")

    monitor_metric = "val_c_index" if args.task_type == "survival" else "val_acc"
    monitor_mode = "max" if monitor_metric in ["val_c_index", "val_acc"] else "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"s_{cur_fold_num}_best",
        save_top_k=1,
        verbose=True,
        monitor=monitor_metric,
        mode=monitor_mode,
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric, patience=20, verbose=True, mode=monitor_mode
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=pl.loggers.TensorBoardLogger(
            save_dir=experiment_base_results_dir, name="logs"
        ),
        log_every_n_steps=10,
    )

    # Training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load best model from checkpoint
    best_checkpoint_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint path: {best_checkpoint_path}")

    # Validate checkpoint path
    if not best_checkpoint_path or not os.path.exists(best_checkpoint_path):
        print(f"Warning: Best checkpoint not found at '{best_checkpoint_path}'")
        print("Using the current model state instead of loading from checkpoint.")
        best_model = model
    else:
        print(f"Loading best model from checkpoint: {best_checkpoint_path}")
        best_model = aegis.load_from_checkpoint(best_checkpoint_path)

    # Validation & Testing with best model
    val_results = trainer.test(best_model, dataloaders=val_loader, verbose=False)[0]
    test_results = trainer.test(best_model, dataloaders=test_loader, verbose=False)[0]

    # Extract metrics
    # Note: PL logs with 'test/' prefix automatically
    if args.task_type == "classification":
        val_auc = val_results.get("test_auc", 0.0)
        val_acc = val_results.get("test_acc", 0.0)
        test_auc = test_results.get("test_auc", 0.0)
        test_acc = test_results.get("test_acc", 0.0)
        return (
            {},
            test_auc,
            val_auc,
            test_acc,
            val_acc,
        )  # Empty dict for patient results for now
    elif args.task_type == "survival":
        val_c_index = val_results.get("test_c_index", 0.0)
        test_c_index = test_results.get("test_c_index", 0.0)
        return {}, test_c_index, val_c_index  # Empty dict for patient results

    return {}, 0.0, 0.0  # Default return
