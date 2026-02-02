"""
Parameter builder classes for constructing configuration dictionaries.
"""

import argparse
import json
from typing import Any, Dict, Optional

from aegis.training.task_utils import is_classification, is_survival


def safe_getattr(args: argparse.Namespace, attr_name: str, default: Any = None) -> Any:
    """
    Safely get attribute from args namespace with default value.

    Args:
        args: Argument namespace
        attr_name: Name of the attribute
        default: Default value if attribute doesn't exist

    Returns:
        Attribute value or default
    """
    return getattr(args, attr_name, default)


def parse_json_arg(
    args: argparse.Namespace, attr_name: str, default: Optional[Any] = None
) -> Optional[Any]:
    """
    Safely parse JSON string from args namespace.

    Args:
        args: Argument namespace
        attr_name: Name of the attribute containing JSON string
        default: Default value if attribute doesn't exist or is empty

    Returns:
        Parsed JSON object or default

    Raises:
        ValueError: If JSON string is invalid
    """
    json_str = safe_getattr(args, attr_name, None)
    if not json_str or json_str == "null":
        return default

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string for {attr_name}: {json_str}") from e


def parse_comma_separated_arg(
    args: argparse.Namespace, attr_name: str, default: Optional[list] = None
) -> Optional[list]:
    """
    Parse comma-separated string from args namespace.

    Args:
        args: Argument namespace
        attr_name: Name of the attribute containing comma-separated string
        default: Default value if attribute doesn't exist or is empty

    Returns:
        List of strings or default
    """
    value = safe_getattr(args, attr_name, None)
    if not value:
        return default
    return value.split(",")


class DataManagerParamsBuilder:
    """Builds parameters for DataManager initialization."""

    @staticmethod
    def build(args: argparse.Namespace) -> Dict[str, Any]:
        """
        Build parameters dictionary for DataManager initialization.

        Args:
            args: Argument namespace

        Returns:
            Dictionary of parameters for DataManager
        """
        params = {
            "task_name": args.task_name,
            "task_type": args.task_type,
            "csv_path": safe_getattr(args, "dataset_csv", None),
            "train_csv": safe_getattr(args, "train_csv", None),
            "val_csv": safe_getattr(args, "val_csv", None),
            "test_csv": safe_getattr(args, "test_csv", None),
            "data_directory": args.data_root_dir,
            "seed": args.seed,
            "verbose": safe_getattr(args, "verbose_data", True),
        }

        # Add task-specific parameters
        if is_classification(args):
            params.update(DataManagerParamsBuilder._build_classification_params(args))
        elif is_survival(args):
            params.update(DataManagerParamsBuilder._build_survival_params(args))

        # Parse label_mapping if provided
        label_mapping = parse_json_arg(args, "label_mapping", None)
        params["label_mapping"] = label_mapping

        return params

    @staticmethod
    def _build_classification_params(args: argparse.Namespace) -> Dict[str, Any]:
        """Build classification-specific parameters."""
        metadata_cols = safe_getattr(args, "metadata_cols", None)
        metadata_columns = (
            [c.strip() for c in metadata_cols.split(",") if c.strip()]
            if metadata_cols
            else None
        )
        return {
            "label_column": safe_getattr(args, "label_col", "label"),
            "patient_id_col_name": safe_getattr(args, "patient_id_col", "case_id"),
            "slide_id_col_name": safe_getattr(args, "slide_id_col", "slide_id"),
            "filter_criteria": parse_json_arg(args, "filter_criteria", None),
            "ignore_labels": parse_comma_separated_arg(args, "ignore_labels", None),
            "patient_label_aggregation": safe_getattr(
                args, "patient_label_aggregation", "max"
            ),
            "shuffle": safe_getattr(args, "shuffle_data", False),
            "split_dir": safe_getattr(args, "split_dir", None),
            "metadata_columns": metadata_columns,
        }

    @staticmethod
    def _build_survival_params(args: argparse.Namespace) -> Dict[str, Any]:
        """Build survival-specific parameters."""
        return {
            "time_column": safe_getattr(args, "time_col", None),
            "event_column": safe_getattr(args, "event_col", None),
            "patient_id_col_name": safe_getattr(args, "patient_id_col", "case_id"),
            "slide_id_col_name": safe_getattr(args, "slide_id_col", "slide_id"),
            "n_bins": safe_getattr(args, "n_bins_survival", 4),
            "filter_dict": parse_json_arg(args, "filter_dict_survival", None),
            "omic_csv_path": safe_getattr(args, "omic_csv", None),
            "omic_patient_id_col": safe_getattr(args, "omic_patient_id_col", "case_id"),
            "apply_sig": safe_getattr(args, "apply_sig_survival", False),
            "signatures_csv_path": safe_getattr(args, "signatures_csv", None),
            "shuffle_slide_data": safe_getattr(
                args, "shuffle_slide_data_survival", False
            ),
        }


class MILDatasetParamsBuilder:
    """Builds parameters for MILDataset creation."""

    @staticmethod
    def build(args: argparse.Namespace) -> Dict[str, Any]:
        """
        Build parameters dictionary for MILDataset creation.

        Args:
            args: Argument namespace

        Returns:
            Dictionary of parameters for MILDataset
        """
        params = {
            "backbone": args.backbone,
            "patch_size": args.patch_size,
            "use_hdf5": safe_getattr(args, "use_hdf5", False),
            "cache_enabled": (safe_getattr(args, "preloading", "no").lower() == "yes"),
            "n_subsamples": safe_getattr(args, "n_subsamples", -1),
            "memmap_bin_path": safe_getattr(args, "memmap_bin_path", None),
            "memmap_json_path": safe_getattr(args, "memmap_json_path", None),
        }

        # Add survival-specific parameters if needed
        if is_survival(args):
            params["mode"] = safe_getattr(args, "survival_mode", "pathomic")

        return params
