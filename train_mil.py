import argparse
import json
import os

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from aegis.data.data_utils import infer_feature_dim_from_data
from aegis.data.dataset_factory import determine_split_directory, get_data_manager
from aegis.parse.cli_parsers import get_mil_args
from aegis.training.pipeline import train_fold
from aegis.utils.environment_setup import (
    create_results_directory,
    log_experiment_details,
    seed_everything,
    setup_device,
)
from aegis.utils.generic_utils import (
    log_results,
    save_pkl,
)


def run_experiment_folds(
    data_manager,
    args: argparse.Namespace,
    experiment_main_results_dir: str,
) -> pd.DataFrame:
    """
    Manages the k-fold cross-validation loop.
    """
    if args.task_type.lower() == "classification":
        metric_keys = [
            "test_auc",
            "val_auc",
            "test_acc",
            "val_acc",
        ]
    elif args.task_type.lower() == "survival":
        metric_keys = [
            "test_c_index",
            "val_c_index",
        ]
    else:
        raise ValueError(f"Unknown task_type: {args.task_type} for defining metrics.")

    all_fold_metrics = {key: [] for key in metric_keys}

    if args.task_type.lower() == "classification":
        data_manager.create_k_fold_splits(
            num_folds=args.k, test_set_size=getattr(args, "test_frac", 0.1)
        )
        num_actual_folds = data_manager.get_number_of_folds()
        if num_actual_folds == 0 and args.k > 0:
            print(
                "No K-folds generated (num_folds=0 in DataManager), running as single train/test split if test_frac > 0."
            )
            if args.k <= 1:
                loop_range = range(1)
                print(f"Running a single train/val/test split (args.k={args.k}).")
            else:
                raise ValueError(
                    f"args.k={args.k} but DataManager created 0 folds. Check data or split logic."
                )
        else:
            loop_range = range(args.k_start, min(args.k_end, num_actual_folds))

    elif args.task_type.lower() == "survival":
        data_manager.create_splits_from_generating_function(
            k=args.k,
            val_num=getattr(args, "val_num_survival", (0.15, 0.15)),
            test_num=getattr(args, "test_num_survival", (0.15, 0.15)),
            label_frac=getattr(args, "label_frac", 1.0),
            custom_test_ids=getattr(args, "custom_test_ids", None),
        )
        loop_range = range(args.k_start, args.k_end)
    else:
        raise ValueError(f"Task type {args.task_type} split creation not defined.")

    for i in loop_range:
        print(f"\n{'=' * 10} Processing Fold: {i} {'=' * 10}")

        if args.task_type.lower() == "classification":
            data_manager.set_current_fold(fold_index=i)
        elif args.task_type.lower() == "survival":
            if not data_manager.set_next_fold_from_generator(
                start_from_fold=i if i == args.k_start else None
            ):
                print(
                    f"SurvivalDataManager's split generator exhausted before reaching fold {i}."
                )
                break

        mil_dataset_params = {
            "backbone": args.backbone,
            "patch_size": args.patch_size,
            "use_hdf5": getattr(args, "use_hdf5", False),
            "cache_enabled": getattr(args, "preloading", "no").lower() == "yes",
            "n_subsamples": getattr(args, "n_subsamples", -1),
            "memmap_bin_path": getattr(args, "memmap_bin_path", None),
            "memmap_json_path": getattr(args, "memmap_json_path", None),
        }

        if args.task_type.lower() == "survival":
            mil_dataset_params["mode"] = getattr(args, "survival_mode", "pathomic")

        train_dataset, val_dataset, test_dataset = data_manager.get_mil_datasets(
            **mil_dataset_params
        )

        if getattr(args, "preloading", "no").lower() == "yes":
            print(f"Preloading data for fold {i}...")
            if train_dataset:
                train_dataset.preload_data()
            if val_dataset:
                val_dataset.preload_data()
            if test_dataset:
                test_dataset.preload_data()

        if getattr(args, "save_splits", False):
            split_file = os.path.join(
                args.split_dir_determined, f"fold_{i}_patient_ids.csv"
            )
            data_manager.save_current_split_patient_ids(split_file)

        fold_results_dir = os.path.join(experiment_main_results_dir, f"fold_{i}")
        os.makedirs(fold_results_dir, exist_ok=True)

        fold_results_tuple = train_fold(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            cur_fold_num=i,
            args=args,
            experiment_base_results_dir=fold_results_dir,
        )

        patient_level_results_dict, *fold_metrics_values = fold_results_tuple

        if len(fold_metrics_values) != len(metric_keys):
            raise ValueError(
                f"Mismatch in metrics from train_fold ({len(fold_metrics_values)}) vs expected ({len(metric_keys)}). "
                f"Task: {args.task_type}. Returned: {fold_metrics_values}"
            )

        for key_idx, metric_value in enumerate(fold_metrics_values):
            metric_name = metric_keys[key_idx]
            all_fold_metrics[metric_name].append(metric_value)

        if patient_level_results_dict:
            fold_results_pkl_path = os.path.join(
                fold_results_dir, "patient_level_results.pkl"
            )
            save_pkl(fold_results_pkl_path, patient_level_results_dict)
            print(
                f"Saved patient-level results for fold {i} to {fold_results_pkl_path}"
            )

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


def main_experiment_runner(args: argparse.Namespace):
    """
    Main function to run the entire experiment.
    """
    _ = setup_device()
    seed_everything(args.seed)

    experiment_main_results_dir = create_results_directory(
        args.results_dir, args.exp_code, args.seed
    )
    args.results_dir = experiment_main_results_dir
    log_experiment_details(args, experiment_main_results_dir)

    args.split_dir_determined = determine_split_directory(
        getattr(args, "split_dir_base", None),
        args.task_name,
        getattr(args, "label_frac", 1.0),
        args.k > 1,
    )
    os.makedirs(args.split_dir_determined, exist_ok=True)

    with SummaryWriter(
        log_dir=os.path.join(experiment_main_results_dir, "summary_all_folds")
    ) as summary_writer:
        args.k_start = max(0, args.k_start)
        args.k_end = args.k if args.k_end == -1 or args.k_end > args.k else args.k_end

        if args.k_start >= args.k_end and args.k > 0:
            print(
                f"Warning: k_start ({args.k_start}) is >= k_end ({args.k_end}). No folds will be run."
            )
            return

        manager_params = {
            "task_name": args.task_name,
            "task_type": args.task_type,
            "csv_path": args.dataset_csv,
            "data_directory": args.data_root_dir,
            "seed": args.seed,
            "verbose": getattr(args, "verbose_data", True),
            "label_column": getattr(args, "label_col", "label"),
            "patient_id_col_name": getattr(args, "patient_id_col", "case_id"),
            "slide_id_col_name": getattr(args, "slide_id_col", "slide_id"),
            "filter_criteria": (
                json.loads(args.filter_criteria)
                if hasattr(args, "filter_criteria") and args.filter_criteria
                else None
            ),
            "ignore_labels": (
                args.ignore_labels.split(",")
                if hasattr(args, "ignore_labels") and args.ignore_labels
                else None
            ),
            "patient_label_aggregation": getattr(
                args,
                "patient_label_aggregation",
                "max",
            ),
            "shuffle": getattr(args, "shuffle_data", False),
            "time_column": getattr(args, "time_col", None),
            "event_column": getattr(args, "event_col", None),
            "n_bins": getattr(args, "n_bins_survival", 4),
            "filter_dict": (
                json.loads(args.filter_dict_survival)
                if hasattr(args, "filter_dict_survival") and args.filter_dict_survival
                else None
            ),
            "omic_csv_path": getattr(args, "omic_csv", None),
            "omic_patient_id_col": getattr(args, "omic_patient_id_col", "case_id"),
            "apply_sig": getattr(args, "apply_sig_survival", False),
            "signatures_csv_path": getattr(args, "signatures_csv", None),
            "shuffle_slide_data": getattr(args, "shuffle_data_survival", False),
            "metadata_columns": (
                [c.strip() for c in args.metadata_cols.split(",") if c.strip()]
                if getattr(args, "metadata_cols", None)
                else None
            ),
        }

        label_mapping_str = getattr(args, "label_mapping", None)
        if label_mapping_str:
            try:
                manager_params["label_mapping"] = json.loads(label_mapping_str)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Invalid JSON string for label_mapping: {label_mapping_str}"
                )
        else:
            manager_params["label_mapping"] = None

        data_manager_instance = get_data_manager(**manager_params)

        args.n_classes = data_manager_instance.num_classes
        args.metadata_dim = getattr(data_manager_instance, "metadata_dim", 0)

        # Infer in_dim from FM features when not provided
        if getattr(args, "in_dim", None) is None and getattr(args, "backbone", None):
            slide_ids = (
                data_manager_instance.slide_data["slide_id"].tolist()[:20]
                if hasattr(data_manager_instance, "slide_data")
                and "slide_id" in data_manager_instance.slide_data.columns
                else []
            )
            inferred = infer_feature_dim_from_data(
                data_manager_instance.data_directory,
                args.backbone,
                slide_ids,
                patch_size=getattr(args, "patch_size", ""),
                use_hdf5=getattr(args, "use_hdf5", False),
            )
            if inferred is not None:
                args.in_dim = inferred

        print(f"DataManager initialized. Number of classes: {args.n_classes}")
        if args.metadata_dim:
            print(f"Multi-modal: metadata_dim={args.metadata_dim}")
        if args.n_classes == 0 and args.task_type.lower() == "classification":
            print(
                "Warning: Number of classes is 0 for classification task. Check data and label mapping."
            )

        print(
            f"Starting experiment: {args.exp_code} with up to {args.k} folds (from {args.k_start} to {args.k_end - 1})"
        )
        overall_results_df = run_experiment_folds(
            data_manager_instance, args, experiment_main_results_dir
        )

        if not overall_results_df.empty:
            log_results(overall_results_df, args, summary_writer)
            print("\nAggregated Results over Folds:")
            print(overall_results_df)
        else:
            print("No results to log as no folds were processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIL Training")
    args = get_mil_args(parser)
    args.backbone = "clam_sb"
    args.task_name = "tcga_ot"

    if not hasattr(args, "task_name"):
        args.task_name = args.task
    if not hasattr(args, "k_fold"):
        args.k_fold = args.k

    main_experiment_runner(args)
    print("Experiment Finished!")
