# aegis/training/cli_runner.py (This replaces your root train_mil.py script)
import argparse
import os
from typing import TYPE_CHECKING

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# Relative imports within the aegis package
from aegis.data.dataset_factory import determine_split_directory, get_data_manager

# IMPORTANT: Ensure aegis/parse/argparse.py is renamed to aegis/parse/cli_parsers.py
from aegis.parse.cli_parsers import get_mil_args
from aegis.training.constants import get_metric_keys_for_task
from aegis.training.fold_processor import process_single_fold
from aegis.training.param_builders import DataManagerParamsBuilder, safe_getattr
from aegis.training.split_manager import SplitManager
from aegis.training.task_utils import is_classification
from aegis.utils.environment_setup import (
    create_results_directory,
    log_experiment_details,
    seed_everything,
    setup_device,
)
from aegis.utils.generic_utils import log_results

if TYPE_CHECKING:
    from aegis.data.classMILDataset import ClassificationDataManager
    from aegis.data.survMILDataset import SurvivalDataManager


def run_experiment_folds(
    data_manager: "ClassificationDataManager | SurvivalDataManager",
    args: argparse.Namespace,
    experiment_main_results_dir: str,
) -> pd.DataFrame:
    """
    Manages the k-fold cross-validation loop.

    Args:
        data_manager: DataManager instance (Classification or Survival)
        args: Argument namespace
        experiment_main_results_dir: Base directory for experiment results

    Returns:
        DataFrame containing metrics for all folds
    """
    # Get metric keys for the task type
    metric_keys = get_metric_keys_for_task(args.task_type)
    all_fold_metrics = {key: [] for key in metric_keys}

    # Create split manager and initialize splits
    split_manager = SplitManager(args)
    loop_range = split_manager.create_splits(data_manager)

    # Process each fold
    for fold_idx in loop_range:
        try:
            patient_level_results_dict, fold_metrics_values = process_single_fold(
                data_manager=data_manager,
                fold_idx=fold_idx,
                args=args,
                experiment_main_results_dir=experiment_main_results_dir,
                metric_keys=metric_keys,
                split_manager=split_manager,
            )

            # Collect metrics
            for key_idx, metric_value in enumerate(fold_metrics_values):
                metric_name = metric_keys[key_idx]
                all_fold_metrics[metric_name].append(metric_value)

        except Exception as e:
            print(f"Error processing fold {fold_idx}: {e}")
            # Continue with next fold or break based on error severity
            # For now, we'll break to avoid silent failures
            break

    # Check if any folds were actually run
    num_folds_run = (
        len(all_fold_metrics[metric_keys[0]])
        if metric_keys and all_fold_metrics[metric_keys[0]]
        else 0
    )
    if num_folds_run == 0:
        print("No folds were successfully processed. Returning empty DataFrame.")
        return pd.DataFrame()

    fold_num_series = list(range(args.k_start, args.k_start + num_folds_run))
    return pd.DataFrame({"fold_num": fold_num_series, **all_fold_metrics})


def run_mil_training_job(args: argparse.Namespace) -> None:
    """
    Main function to run the entire MIL training experiment.

    Args:
        args: Argument namespace containing all training configuration
    """
    # Setup environment
    _ = setup_device()
    seed_everything(args.seed)

    # Create results directory
    experiment_main_results_dir = create_results_directory(
        args.results_dir, args.exp_code, args.seed
    )
    args.results_dir = experiment_main_results_dir  # Update args for train_fold
    log_experiment_details(args, experiment_main_results_dir)

    # Determine split directory
    args.split_dir_determined = determine_split_directory(
        safe_getattr(args, "split_dir_base", None),
        args.task_name,
        safe_getattr(args, "label_frac", 1.0),
        args.k > 1,  # k_fold is true if num_folds > 1
    )
    os.makedirs(args.split_dir_determined, exist_ok=True)

    with SummaryWriter(
        log_dir=os.path.join(experiment_main_results_dir, "summary_all_folds")
    ) as summary_writer:
        # Validate and set fold range
        args.k_start = max(0, args.k_start)
        args.k_end = args.k if args.k_end == -1 or args.k_end > args.k else args.k_end

        if args.k_start >= args.k_end and args.k > 0:  # k_end is exclusive
            print(
                f"Warning: k_start ({args.k_start}) is >= k_end ({args.k_end}). "
                "No folds will be run."
            )
            return

        # Build DataManager parameters and create instance
        manager_params = DataManagerParamsBuilder.build(args)
        data_manager_instance = get_data_manager(**manager_params)

        # Get n_classes and metadata_dim from DataManager and set in args for train_fold
        args.n_classes = data_manager_instance.num_classes
        args.metadata_dim = getattr(data_manager_instance, "metadata_dim", 0)
        print(f"DataManager initialized. Number of classes: {args.n_classes}")
        if args.metadata_dim:
            print(f"Multi-modal: metadata_dim={args.metadata_dim}")
        if args.n_classes == 0 and is_classification(args):
            print(
                "Warning: Number of classes is 0 for classification task. "
                "Check data and label mapping."
            )

        print(
            f"Starting experiment: {args.exp_code} with up to {args.k} folds "
            f"(from {args.k_start} to {args.k_end - 1})"
        )

        # Run experiment folds
        overall_results_df = run_experiment_folds(
            data_manager_instance, args, experiment_main_results_dir
        )

        # Log and display results
        if not overall_results_df.empty:
            log_results(overall_results_df, args, summary_writer)
            print("\nAggregated Results over Folds:")
            print(overall_results_df)
        else:
            print("No results to log as no folds were processed.")


if __name__ == "__main__":
    # This block is for direct execution during development/testing outside of package
    # For package usage, `aegis.cli:train_mil_main` will be called.
    args = get_mil_args()
    # Potentially add more argument validation or default setting here if needed
    if not hasattr(args, "task_name"):
        args.task_name = args.task
    if not hasattr(args, "k_fold"):
        args.k_fold = args.k
    run_mil_training_job(args)
    print("Experiment Finished!")
