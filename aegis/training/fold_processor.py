"""
Functions for processing individual folds during training.
"""

import argparse
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from aegis.training.param_builders import MILDatasetParamsBuilder, safe_getattr
from aegis.training.pipeline import train_fold
from aegis.training.split_manager import SplitManager
from aegis.utils.generic_utils import save_pkl

if TYPE_CHECKING:
    from aegis.data.classMILDataset import ClassificationDataManager
    from aegis.data.survMILDataset import SurvivalDataManager


def prepare_fold_datasets(
    data_manager: "ClassificationDataManager | SurvivalDataManager",
    args: argparse.Namespace,
    fold_idx: int,
) -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """
    Prepare train/val/test datasets for a fold.

    Args:
        data_manager: DataManager instance
        args: Argument namespace
        fold_idx: Index of the current fold

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Build MILDataset parameters
    mil_dataset_params = MILDatasetParamsBuilder.build(args)

    # Get datasets from data manager
    train_dataset, val_dataset, test_dataset = data_manager.get_mil_datasets(
        **mil_dataset_params
    )

    return train_dataset, val_dataset, test_dataset


def preload_datasets_if_requested(
    train_dataset: Optional[object],
    val_dataset: Optional[object],
    test_dataset: Optional[object],
    fold_idx: int,
    args: argparse.Namespace,
) -> None:
    """
    Preload datasets if preloading is enabled.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        fold_idx: Index of the current fold
        args: Argument namespace
    """
    if safe_getattr(args, "preloading", "no").lower() == "yes":
        print(f"Preloading data for fold {fold_idx}...")
        for dataset in [train_dataset, val_dataset, test_dataset]:
            if dataset and hasattr(dataset, "preload_data"):
                dataset.preload_data()


def save_split_if_requested(
    data_manager: "ClassificationDataManager | SurvivalDataManager",
    args: argparse.Namespace,
    fold_idx: int,
) -> None:
    """
    Save current split patient IDs if requested.

    Args:
        data_manager: DataManager instance
        args: Argument namespace
        fold_idx: Index of the current fold
    """
    if safe_getattr(args, "save_splits", False):
        split_file = os.path.join(
            args.split_dir_determined, f"fold_{fold_idx}_patient_ids.csv"
        )
        if hasattr(data_manager, "save_current_split_patient_ids"):
            data_manager.save_current_split_patient_ids(split_file)


def process_single_fold(
    data_manager: "ClassificationDataManager | SurvivalDataManager",
    fold_idx: int,
    args: argparse.Namespace,
    experiment_main_results_dir: str,
    metric_keys: List[str],
    split_manager: SplitManager,
) -> Tuple[Dict, List[float]]:
    """
    Process a single fold: prepare datasets, train, and collect results.

    Args:
        data_manager: DataManager instance
        fold_idx: Index of the current fold
        args: Argument namespace
        experiment_main_results_dir: Base directory for experiment results
        metric_keys: List of metric keys expected from train_fold
        split_manager: SplitManager instance

    Returns:
        Tuple of (patient_level_results_dict, list of metric values)

    Raises:
        ValueError: If metrics from train_fold don't match expected metric_keys
    """
    print(f"\n{'=' * 10} Processing Fold: {fold_idx} {'=' * 10}")

    # Set the current fold in the DataManager
    if not split_manager.set_current_fold(data_manager, fold_idx):
        # For survival, if set_current_fold returns False, skip this fold
        return {}, []

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_fold_datasets(
        data_manager, args, fold_idx
    )

    # Preload data if requested
    preload_datasets_if_requested(
        train_dataset, val_dataset, test_dataset, fold_idx, args
    )

    # Save current split patient IDs if requested
    save_split_if_requested(data_manager, args, fold_idx)

    # Get fold-specific results directory
    fold_results_dir = os.path.join(experiment_main_results_dir, f"fold_{fold_idx}")
    os.makedirs(fold_results_dir, exist_ok=True)

    # Train the model for this fold
    fold_results_tuple = train_fold(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        cur_fold_num=fold_idx,
        args=args,
        experiment_base_results_dir=fold_results_dir,
    )

    # Unpack results
    patient_level_results_dict, *fold_metrics_values = fold_results_tuple

    # Validate metrics
    if len(fold_metrics_values) != len(metric_keys):
        raise ValueError(
            f"Mismatch in metrics from train_fold ({len(fold_metrics_values)}) "
            f"vs expected ({len(metric_keys)}). "
            f"Task: {args.task_type}. Returned: {fold_metrics_values}"
        )

    # Save patient-level results if available
    if patient_level_results_dict:
        fold_results_pkl_path = os.path.join(
            fold_results_dir, "patient_level_results.pkl"
        )
        save_pkl(fold_results_pkl_path, patient_level_results_dict)
        print(
            f"Saved patient-level results for fold {fold_idx} to {fold_results_pkl_path}"
        )

    return patient_level_results_dict, fold_metrics_values
