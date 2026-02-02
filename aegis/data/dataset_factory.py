import os
from typing import Any, Dict, Optional, Union

# Assuming your new dataset classes are in these locations
from aegis.data.classMILDataset import ClassificationDataManager
from aegis.data.survMILDataset import SurvivalDataManager

# Keep supported tasks as in your original, or manage elsewhere
SUPPORTED_TASKS = ["TGCT", "BRCA", "COAD", "UCEC", "LUAD"]


def get_data_manager(
    task_name: str,
    task_type: str,
    csv_path: Optional[str],
    data_directory: Union[str, Dict[str, str]],
    seed: int = 7,
    verbose: bool = True,
    **kwargs: Any,  # Catch-all for other specific params
) -> Union[ClassificationDataManager, SurvivalDataManager]:
    """
    Factory function to create a DataManager instance for a given task and type.

    Args for Classification (via kwargs):
        label_column (str): Name of the column in CSV containing original labels.
        label_mapping (Dict[str, int]): Mapping from string labels to integer classes.
        patient_id_col_name (str): Column name for patient IDs. Default 'case_id'.
        slide_id_col_name (str): Column name for slide IDs. Default 'slide_id'.
        filter_criteria (Dict): For filtering rows in CSV.
        ignore_labels (List[str]): String labels to ignore.
        patient_label_aggregation (str): 'max' or 'majority'.
        shuffle (bool): Whether to shuffle loaded data. Default False. (for ClassificationDataManager)

    Args for Survival (via kwargs):
        time_column (str): Name of the column for survival time.
        event_column (str): Name of the column for event status (0=censored, 1=event).
        patient_id_col_name (str): Column name for patient IDs. Default 'case_id'.
        slide_id_col_name (str): Column name for slide IDs. Default 'slide_id'.
        n_bins (int): Number of bins for discretizing survival time. Default 4.
        filter_dict (Dict): For filtering rows in CSV.
        omic_csv_path (str): Path to CSV with omic features.
        omic_patient_id_col (str): Patient ID column in omic CSV.
        apply_sig (bool): For coattn mode signatures.
        signatures_csv_path (str): Path to signatures CSV for coattn.
        shuffle_slide_data (bool): Shuffle initial slide data. Default False. (for SurvivalDataManager)
    """
    task_name_upper = task_name.upper()
    if task_name_upper not in SUPPORTED_TASKS:
        # Consider if this check is still needed if task_type drives logic
        print(
            f"Warning: Task '{task_name}' not in predefined SUPPORTED_TASKS, but proceeding based on task_type."
        )

    if csv_path and not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV file not found: {csv_path}")

    # Extract split CSVs from kwargs
    train_csv = kwargs.get("train_csv")
    val_csv = kwargs.get("val_csv")
    test_csv = kwargs.get("test_csv")

    if not csv_path and not train_csv:
        raise ValueError("Either csv_path or train_csv must be provided.")

    if task_type.lower() == "classification":
        # Extract classification-specific args from kwargs with defaults
        cls_kwargs = {
            "label_column": kwargs.get("label_column", "label"),
            "label_mapping": kwargs.get("label_mapping"),  # Allow None, DM can infer
            "patient_id_col_name": kwargs.get("patient_id_col_name", "case_id"),
            "slide_id_col_name": kwargs.get("slide_id_col_name", "slide_id"),
            "shuffle": kwargs.get(
                "shuffle", False
            ),  # Specific to ClassificationDataManager's initial load
            "filter_criteria": kwargs.get("filter_criteria"),
            "ignore_labels": kwargs.get("ignore_labels"),
            "patient_stratification": kwargs.get(
                "patient_stratification", False
            ),  # Legacy, not directly used by DM for len
            "patient_label_aggregation": kwargs.get("patient_label_aggregation", "max"),
            "metadata_columns": kwargs.get("metadata_columns"),
        }
        return ClassificationDataManager(
            csv_path=csv_path,
            data_directory=data_directory,
            random_seed=seed,
            verbose=verbose,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            **cls_kwargs,
        )
    elif task_type.lower() == "survival":
        # Extract survival-specific args from kwargs with defaults
        surv_kwargs = {
            "time_column": kwargs.get("time_column"),
            "event_column": kwargs.get("event_column"),
            "patient_id_col_name": kwargs.get("patient_id_col_name", "case_id"),
            "slide_id_col_name": kwargs.get("slide_id_col_name", "slide_id"),
            "n_bins": kwargs.get("n_bins", 4),
            "shuffle_slide_data": kwargs.get(
                "shuffle_slide_data", False
            ),  # Specific to SurvivalDataManager
            "filter_dict": kwargs.get("filter_dict"),
            "eps": kwargs.get("eps", 1e-6),
            "omic_csv_path": kwargs.get("omic_csv_path"),
            "omic_patient_id_col": kwargs.get("omic_patient_id_col", "case_id"),
            "apply_sig": kwargs.get("apply_sig", False),
            "signatures_csv_path": kwargs.get("signatures_csv_path"),
        }
        if not surv_kwargs["time_column"] or not surv_kwargs["event_column"]:
            raise ValueError(
                "time_column and event_column must be provided for survival tasks."
            )

        return SurvivalDataManager(
            csv_path=csv_path,
            data_directory=data_directory,
            random_seed=seed,
            verbose=verbose,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            **surv_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported task_type: '{task_type}'. Must be 'classification' or 'survival'."
        )


def determine_split_directory(  # This function seems more for organizing output/split files
    base_split_dir: Optional[str],
    task_name: str,
    label_frac: float = 1.0,
    k_fold: bool = True,
) -> str:
    """
    Determines a directory path for storing/loading data splits or results.
    (This might be used externally to the DataManager for saving outputs).
    """
    if base_split_dir is None:
        # Example: splits/TGCT/label_frac_100_kfold or splits/TGCT/label_frac_10_holdout
        split_subdir = f"{task_name.upper()}"
        # Suffix based on k_fold and label_frac (if not full dataset)
        suffix = "_kfold" if k_fold else "_holdout"
        if label_frac < 1.0:  # Only add label_frac if it's a subset
            split_subdir += f"_label_frac_{int(label_frac * 100)}"

        determined_dir = os.path.join("results_or_splits", split_subdir + suffix)
    else:
        determined_dir = base_split_dir

    # Ensure directory exists if it's for saving
    # os.makedirs(determined_dir, exist_ok=True) # Uncomment if you want auto-creation
    return determined_dir
