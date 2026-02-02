from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

# Import memmap dataset for conditional use
try:
    from aegis.data.memmapMILDataset import MemmapDataset
except ImportError:
    MemmapDataset = None

# Worker-local HDF5 file handle cache to avoid opening/closing files repeatedly
# This is shared across all dataset instances in the same worker process
_worker_hdf5_cache: Dict[str, h5py.File] = {}
_worker_hdf5_cache_lock = threading.Lock()


class ClassificationDataManager:
    def __init__(
        self,
        csv_path: Optional[str] = None,
        data_directory: Union[str, Dict[str, str]] = None,
        label_column: str = "label",  # Actual column name in CSV for labels
        label_mapping: Optional[Dict[str, int]] = None,
        patient_id_col_name: str = "case_id",  # Actual col name in CSV for patient ID
        slide_id_col_name: str = "slide_id",  # Actual col name in CSV for slide ID
        shuffle: bool = False,
        random_seed: int = 7,
        verbose: bool = True,
        filter_criteria: Optional[Dict[str, List[str]]] = None,
        ignore_labels: Optional[List[str]] = None,  # Original string labels to ignore
        patient_stratification: bool = False,  # If true, __len__ uses patient_data
        patient_label_aggregation: str = "max",  # 'max' or 'majority'
        split_dir: Optional[str] = None,
        train_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        site_column: Optional[str] = None,  # Column name for site information
    ):
        self.csv_path = csv_path
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv_path = test_csv
        self.data_directory = data_directory
        self.provided_label_column = label_column
        self.label_mapping = label_mapping or {}
        self.patient_id_col_name = patient_id_col_name
        self.slide_id_col_name = slide_id_col_name
        self.random_seed = random_seed
        self.verbose = verbose
        self.patient_stratification = (
            patient_stratification  # Used by __len__ if this class were a Dataset
        )
        self.split_dir = split_dir
        self.site_column = site_column
        self.site_mapping: Dict[str, int] = {}

        self.train_patient_ids: Optional[List[str]] = None
        self.val_patient_ids: Optional[List[str]] = None
        self.test_patient_ids: Optional[List[str]] = None

        self.train_slide_indices: Optional[List[int]] = None
        self.val_slide_indices: Optional[List[int]] = None
        self.test_slide_indices: Optional[List[int]] = None

        self.kfold_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.train_val_patient_data_for_kfold: Optional[pd.DataFrame] = None

        # Load and preprocess slide data
        if csv_path is not None:
            try:
                raw_slide_data = pd.read_csv(csv_path)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"CSV file not found: {csv_path}") from e
        elif train_csv is not None:
            # Load separate CSVs
            try:
                train_df = pd.read_csv(train_csv)
                self.train_patient_ids = (
                    train_df[self.patient_id_col_name].unique().tolist()
                )

                dfs = [train_df]
                if val_csv:
                    val_df = pd.read_csv(val_csv)
                    self.val_patient_ids = (
                        val_df[self.patient_id_col_name].unique().tolist()
                    )
                    dfs.append(val_df)
                else:
                    self.val_patient_ids = []

                if test_csv:
                    test_df = pd.read_csv(test_csv)
                    self.test_patient_ids = (
                        test_df[self.patient_id_col_name].unique().tolist()
                    )
                    dfs.append(test_df)
                else:
                    self.test_patient_ids = []

                raw_slide_data = pd.concat(dfs, ignore_index=True)

            except FileNotFoundError as e:
                raise FileNotFoundError(f"One of the split CSV files not found") from e
        else:
            raise ValueError("Either csv_path or train_csv must be provided.")

        # Standardize column names internally
        self.slide_data = self._rename_cols(raw_slide_data)

        self.slide_data = self._filter_data(self.slide_data, filter_criteria)
        self.slide_data = self._prepare_labels(self.slide_data, ignore_labels)

        if not self.label_mapping:  # Infer mapping if not provided
            unique_labels = self.slide_data["label_str"].unique()
            self.label_mapping = {
                label: i for i, label in enumerate(sorted(unique_labels))
            }
            if verbose:
                print(f"Inferred label_mapping: {self.label_mapping}")

        # Apply the final mapping after potential inference
        self.slide_data["label"] = self.slide_data["label_str"].map(self.label_mapping)
        # Drop rows where mapping resulted in NaN (e.g. label_str not in inferred map)
        self.slide_data.dropna(subset=["label"], inplace=True)
        self.slide_data["label"] = self.slide_data["label"].astype(int)

        # Process site column if provided
        if self.site_column:
            if self.site_column not in self.slide_data.columns:
                raise ValueError(f"Site column '{self.site_column}' not found in CSV.")

            # Create site mapping
            unique_sites = sorted(
                self.slide_data[self.site_column].astype(str).unique()
            )
            self.site_mapping = {site: i for i, site in enumerate(unique_sites)}

            # Map sites to integers
            self.slide_data["site_id"] = (
                self.slide_data[self.site_column].astype(str).map(self.site_mapping)
            )
            # Handle potential NaNs if any (though we built map from unique values)
            if self.slide_data["site_id"].isnull().any():
                raise ValueError("NaN values found in site column mapping.")
            self.slide_data["site_id"] = self.slide_data["site_id"].astype(int)

            if verbose:
                print(f"Inferred site_mapping: {self.site_mapping}")
                print(f"Number of sites: {len(self.site_mapping)}")

        self.num_classes = len(set(self.label_mapping.values()))
        if self.num_classes == 0 and len(self.slide_data) > 0:
            raise ValueError(
                "Number of classes is 0. Check label_column and label_mapping. "
                f"Unique labels found: {self.slide_data['label_str'].unique()}"
            )

        if shuffle:
            self.slide_data = self.slide_data.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)

        self._aggregate_patient_data(patient_label_aggregation)
        self._prepare_class_indices()

        if verbose:
            self._print_summary()

        # If separate CSVs were provided, we need to set the indices now that processing is done
        if csv_path is None and train_csv is not None:
            # Re-derive indices based on patient IDs (which we stored) and the processed slide_data
            # Note: slide_data might have been filtered, so we need to be careful

            # Update patient IDs based on what remains in slide_data
            available_patients = set(self.slide_data["case_id"].unique())

            self.train_patient_ids = [
                pid for pid in self.train_patient_ids if pid in available_patients
            ]
            self.val_patient_ids = [
                pid for pid in self.val_patient_ids if pid in available_patients
            ]
            self.test_patient_ids = [
                pid for pid in self.test_patient_ids if pid in available_patients
            ]

            self.train_slide_indices = self.slide_data[
                self.slide_data["case_id"].isin(self.train_patient_ids)
            ].index.tolist()
            self.val_slide_indices = self.slide_data[
                self.slide_data["case_id"].isin(self.val_patient_ids)
            ].index.tolist()
            self.test_slide_indices = self.slide_data[
                self.slide_data["case_id"].isin(self.test_patient_ids)
            ].index.tolist()

            # Create a dummy kfold_splits to make set_current_fold work or just set it directly
            # Since we have fixed splits, we can treat it as a single fold
            self.kfold_splits = [(self.train_patient_ids, self.val_patient_ids)]

    def _rename_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if (
            self.patient_id_col_name != "case_id"
            and self.patient_id_col_name in df.columns
        ):
            df.rename(columns={self.patient_id_col_name: "case_id"}, inplace=True)
        if (
            self.slide_id_col_name != "slide_id"
            and self.slide_id_col_name in df.columns
        ):
            df.rename(columns={self.slide_id_col_name: "slide_id"}, inplace=True)
        if "case_id" not in df.columns:
            raise ValueError(
                f"Patient ID column '{self.patient_id_col_name}' (expected 'case_id') not found in CSV."
            )
        if "slide_id" not in df.columns:
            raise ValueError(
                f"Slide ID column '{self.slide_id_col_name}' (expected 'slide_id') not found in CSV."
            )
        return df

    def _filter_data(
        self, data: pd.DataFrame, filter_criteria: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        if filter_criteria:
            mask = pd.Series(True, index=data.index)
            for column, values in filter_criteria.items():
                if column not in data.columns:
                    print(f"Warning: Filter column '{column}' not found in data.")
                    continue
                mask &= data[column].isin(values)
            data = data[mask].reset_index(drop=True)
        return data

    def _prepare_labels(
        self, data: pd.DataFrame, ignore_labels_str: Optional[List[str]]
    ) -> pd.DataFrame:
        data = data.copy()
        if self.provided_label_column not in data.columns:
            raise ValueError(
                f"Provided label column '{self.provided_label_column}' not found in CSV."
            )

        # Keep original string labels in 'label_str' for mapping and ignoring
        data["label_str"] = data[self.provided_label_column].astype(str)

        if ignore_labels_str:
            data = data[~data["label_str"].isin(ignore_labels_str)].reset_index(
                drop=True
            )

        # Integer mapping will be applied after potential inference in __init__
        return data

    def _aggregate_patient_data(self, aggregation_method: str = "max") -> None:
        if (
            "case_id" not in self.slide_data.columns
            or "label" not in self.slide_data.columns
        ):
            print(
                "Warning: 'case_id' or 'label' not fully prepared for patient aggregation."
            )
            self.patient_data = pd.DataFrame(columns=["case_id", "label"])
            return

        patients = self.slide_data["case_id"].unique()
        patient_labels_agg = []

        for patient_id in patients:
            labels = self.slide_data.loc[
                self.slide_data["case_id"] == patient_id, "label"
            ].values
            if len(labels) == 0:
                continue  # Should not happen if patient_id is from slide_data

            if aggregation_method == "max":
                aggregated_label = labels.max()
            elif aggregation_method == "majority":
                aggregated_label = mode(labels, keepdims=True).mode[0]
            else:
                raise ValueError(
                    f"Invalid patient_label_aggregation: {aggregation_method}"
                )
            patient_labels_agg.append(aggregated_label)

        self.patient_data = pd.DataFrame(
            {"case_id": patients, "label": patient_labels_agg}
        )

    def _prepare_class_indices(self) -> None:
        if self.num_classes > 0:
            self.patient_class_indices = [
                np.where(self.patient_data["label"] == cls_label)[0]
                for cls_label in range(self.num_classes)
            ]
            self.slide_class_indices = [
                np.where(self.slide_data["label"] == cls_label)[0]
                for cls_label in range(self.num_classes)
            ]
        else:
            self.patient_class_indices = []
            self.slide_class_indices = []

    def _print_summary(self) -> None:
        print("--- Classification DataManager Summary ---")
        print(f"CSV Path: {self.csv_path}")
        print(f"Label Column (original): {self.provided_label_column}")
        print(f"Patient ID Column (original): {self.patient_id_col_name}")
        print(f"Slide ID Column (original): {self.slide_id_col_name}")
        print(f"Label Mapping: {self.label_mapping}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Total unique slides: {self.slide_data['slide_id'].nunique()}")
        print(f"Total unique patients: {self.patient_data['case_id'].nunique()}")

        if self.num_classes > 0:
            print("\nSlide-level counts (after mapping):")
            print(self.slide_data["label"].value_counts(sort=False))
            print("\nPatient-level counts (after aggregation and mapping):")
            print(self.patient_data["label"].value_counts(sort=False))
        else:
            print("\nNo classes defined or no data loaded.")
        print("----------------------------------------")

    def _load_splits_from_dir(self, num_folds: int) -> bool:
        if not self.split_dir or not os.path.isdir(self.split_dir):
            return False

        self.kfold_splits = []
        for i in range(num_folds):
            split_file = os.path.join(self.split_dir, f"splits_{i}.csv")
            if not os.path.isfile(split_file):
                print(f"Warning: Split file not found for fold {i}: {split_file}")
                continue

            df = pd.read_csv(split_file)
            train_ids = df["train_patient_id"].dropna().tolist()
            val_ids = df["val_patient_id"].dropna().tolist()
            # The test set is loaded once and assumed to be consistent across folds
            if self.test_patient_ids is None and "test_patient_id" in df:
                self.test_patient_ids = df["test_patient_id"].dropna().tolist()

            # Store patient IDs for each fold
            self.kfold_splits.append((train_ids, val_ids))

        if not self.kfold_splits:
            print(
                f"Warning: No split files found in {self.split_dir}. Falling back to automatic splitting."
            )
            return False

        if self.verbose:
            print(
                f"Successfully loaded {len(self.kfold_splits)} splits from {self.split_dir}"
            )
            if self.test_patient_ids:
                print(f"Loaded {len(self.test_patient_ids)} test patient IDs.")
        return True

    def create_k_fold_splits(
        self, num_folds: int = 5, test_set_size: float = 0.1
    ) -> None:
        if self.kfold_splits is not None:
            if self.verbose:
                print(
                    "Splits already set (e.g. from separate CSVs or loaded). Skipping creation."
                )
            return

        if self.patient_data.empty:
            raise ValueError("Patient data is empty. Cannot create splits.")

        # Attempt to load splits from directory first
        if self._load_splits_from_dir(num_folds):
            # Test patient IDs are already set by _load_splits_from_dir
            # The kfold_splits now contains (train_ids, val_ids) tuples
            return

        # Create initial test set from patients
        if test_set_size > 0:
            train_val_patients_df, test_patients_df = train_test_split(
                self.patient_data,
                test_size=test_set_size,
                stratify=self.patient_data["label"],
                random_state=self.random_seed,
            )
            self.test_patient_ids = test_patients_df["case_id"].tolist()
        else:
            train_val_patients_df = self.patient_data.copy()
            self.test_patient_ids = []

        if num_folds > 1 and not train_val_patients_df.empty:
            skf = StratifiedKFold(
                n_splits=num_folds, shuffle=True, random_state=self.random_seed
            )
            # Ensure indices are reset for iloc to work correctly with skf.split
            self.train_val_patient_data_for_kfold = train_val_patients_df.reset_index(
                drop=True
            )
            # This kfold_splits contains indices, not patient IDs
            self.kfold_splits_indices = list(
                skf.split(
                    self.train_val_patient_data_for_kfold,
                    self.train_val_patient_data_for_kfold["label"],
                )
            )
        elif num_folds == 1 and not train_val_patients_df.empty:
            # If num_folds is 1, treat it as a single train/val split from the remaining data
            # We need to split train_val_patients_df into actual train and val
            train_patients_df, val_patients_df = train_test_split(
                train_val_patients_df,
                test_size=0.2,  # Default 20% for validation if not k-folding
                stratify=train_val_patients_df["label"],
                random_state=self.random_seed,
            )
            self.train_patient_ids = train_patients_df["case_id"].tolist()
            self.val_patient_ids = val_patients_df["case_id"].tolist()
            self.kfold_splits = None
            self.kfold_splits_indices = None
        else:
            # If no k-folds (num_folds <= 0) or empty df, all train_val_patients become train, and val is empty
            self.train_patient_ids = train_val_patients_df["case_id"].tolist()
            self.val_patient_ids = []
            self.kfold_splits = None
            self.kfold_splits_indices = None

        if self.verbose:
            print(
                f"Created {num_folds}-fold splits with test set size {test_set_size}."
            )
            print(f"Total patients for K-Fold Train/Val: {len(train_val_patients_df)}")
            print(f"Total patients for Test: {len(self.test_patient_ids)}")

    def set_current_fold(self, fold_index: int = 0) -> None:
        # Case 1: Splits were loaded from a directory
        if self.kfold_splits and isinstance(self.kfold_splits[0][0], list):
            if fold_index >= len(self.kfold_splits):
                raise ValueError(
                    f"Fold index {fold_index} is out of bounds for {len(self.kfold_splits)} loaded splits."
                )

            self.train_patient_ids, self.val_patient_ids = self.kfold_splits[fold_index]
            # self.test_patient_ids is already set

        # Case 2: No k-folds were generated (e.g., k=0 or k=1), simple train/val/test split
        elif (
            self.kfold_splits is None
            or not hasattr(self, "kfold_splits_indices")
            or self.kfold_splits_indices is None
        ) and self.train_patient_ids is not None:
            pass  # Patient IDs are already set in create_k_fold_splits

        # Case 3: Splits were generated by StratifiedKFold (original logic)
        elif (
            hasattr(self, "kfold_splits_indices")
            and self.kfold_splits_indices is not None
        ):
            if self.train_val_patient_data_for_kfold is None:
                raise ValueError(
                    "K-Fold splits have not been created. Call create_k_fold_splits() first."
                )
            if fold_index >= len(self.kfold_splits_indices):
                raise ValueError(
                    f"Fold index {fold_index} is out of bounds for {len(self.kfold_splits_indices)} folds."
                )

            train_patient_indices, val_patient_indices = self.kfold_splits_indices[
                fold_index
            ]

            train_patients_df = self.train_val_patient_data_for_kfold.iloc[
                train_patient_indices
            ]
            val_patients_df = self.train_val_patient_data_for_kfold.iloc[
                val_patient_indices
            ]

            self.train_patient_ids = train_patients_df["case_id"].tolist()
            self.val_patient_ids = val_patients_df["case_id"].tolist()

        else:
            raise ValueError("Split state is inconsistent. Could not set fold.")

        # Map patient IDs to slide indices
        self.train_slide_indices = (
            self.slide_data[
                self.slide_data["case_id"].isin(self.train_patient_ids)
            ].index.tolist()
            if self.train_patient_ids
            else []
        )
        self.val_slide_indices = (
            self.slide_data[
                self.slide_data["case_id"].isin(self.val_patient_ids)
            ].index.tolist()
            if self.val_patient_ids
            else []
        )
        self.test_slide_indices = (
            self.slide_data[
                self.slide_data["case_id"].isin(self.test_patient_ids)
            ].index.tolist()
            if self.test_patient_ids
            else []
        )

        if self.verbose:
            print(f"Set to fold {fold_index}:")
            print(
                f"  Train patients: {len(self.train_patient_ids)}, Train slides: {len(self.train_slide_indices)}"
            )
            print(
                f"  Val patients: {len(self.val_patient_ids)}, Val slides: {len(self.val_slide_indices)}"
            )
            print(
                f"  Test patients: {len(self.test_patient_ids)}, Test slides: {len(self.test_slide_indices)}"
            )

    def get_mil_datasets(
        self,
        backbone: str,
        patch_size: str = "",
        use_hdf5: bool = False,
        cache_enabled: bool = False,
        n_subsamples: int = -1,
        memmap_bin_path: Optional[str] = None,  # Path to memmap binary file
        memmap_json_path: Optional[str] = None,  # Path to memmap index JSON file
    ) -> Tuple[
        Optional[WSIMILDataset], Optional[WSIMILDataset], Optional[WSIMILDataset]
    ]:
        if (
            self.train_slide_indices is None
            or self.val_slide_indices is None
            or self.test_slide_indices is None
        ):
            raise ValueError(
                "Splits/fold not set. Call create_k_fold_splits() and then set_current_fold() first."
            )

        train_df_split = self.slide_data.iloc[self.train_slide_indices].reset_index(
            drop=True
        )
        val_df_split = self.slide_data.iloc[self.val_slide_indices].reset_index(
            drop=True
        )
        test_df_split = self.slide_data.iloc[self.test_slide_indices].reset_index(
            drop=True
        )

        # Check if memmap paths are provided
        use_memmap = memmap_bin_path is not None and memmap_json_path is not None

        if use_memmap:
            # Use MemmapDataset
            if MemmapDataset is None:
                raise ImportError(
                    "MemmapDataset not available. "
                    "Ensure aegis.data.memmapMILDataset is properly imported."
                )

            train_dataset = (
                MemmapDataset(
                    bin_path=memmap_bin_path,
                    json_path=memmap_json_path,
                    slide_data_df=train_df_split,
                    n_subsamples=n_subsamples if n_subsamples > 0 else 2048,
                )
                if not train_df_split.empty
                else None
            )
            val_dataset = (
                MemmapDataset(
                    bin_path=memmap_bin_path,
                    json_path=memmap_json_path,
                    slide_data_df=val_df_split,
                    n_subsamples=-1,
                )
                if not val_df_split.empty
                else None
            )
            test_dataset = (
                MemmapDataset(
                    bin_path=memmap_bin_path,
                    json_path=memmap_json_path,
                    slide_data_df=test_df_split,
                    n_subsamples=-1,
                )
                if not test_df_split.empty
                else None
            )
        else:
            # Use original WSIMILDataset
            common_params = {
                "data_directory": self.data_directory,
                "num_classes": self.num_classes,
                "backbone": backbone,
                "patch_size": patch_size,
                "use_hdf5": use_hdf5,
                "cache_enabled": cache_enabled,
                "site_column": "site_id" if self.site_column else None,
            }

            train_params = common_params.copy()
            train_params["n_subsamples"] = n_subsamples

            eval_params = common_params.copy()
            eval_params["n_subsamples"] = -1

            train_dataset = (
                WSIMILDataset(slide_data_df=train_df_split, **train_params)
                if not train_df_split.empty
                else None
            )
            val_dataset = (
                WSIMILDataset(slide_data_df=val_df_split, **eval_params)
                if not val_df_split.empty
                else None
            )
            test_dataset = (
                WSIMILDataset(slide_data_df=test_df_split, **eval_params)
                if not test_df_split.empty
                else None
            )

        return train_dataset, val_dataset, test_dataset

    def get_number_of_folds(self) -> int:
        return len(self.kfold_splits) if self.kfold_splits is not None else 0

    def save_current_split_patient_ids(self, filename: str) -> None:
        if (
            self.train_patient_ids is None
            or self.val_patient_ids is None
            or self.test_patient_ids is None
        ):
            raise ValueError("Splits not set. Call set_current_fold() first.")

        # Pad lists to the same length for DataFrame creation
        max_len = max(
            len(self.train_patient_ids or []),
            len(self.val_patient_ids or []),
            len(self.test_patient_ids or []),
        )

        train_ids_padded = (self.train_patient_ids or []) + [None] * (
            max_len - len(self.train_patient_ids or [])
        )
        val_ids_padded = (self.val_patient_ids or []) + [None] * (
            max_len - len(self.val_patient_ids or [])
        )
        test_ids_padded = (self.test_patient_ids or []) + [None] * (
            max_len - len(self.test_patient_ids or [])
        )

        splits_df = pd.DataFrame(
            {
                "train_patient_id": train_ids_padded,
                "val_patient_id": val_ids_padded,
                "test_patient_id": test_ids_padded,
            }
        )
        splits_df.to_csv(filename, index=False)
        if self.verbose:
            print(f"Current split patient IDs saved to {filename}")

    def summarize_current_splits(
        self, return_summary_df: bool = False
    ) -> Optional[pd.DataFrame]:
        if self.train_slide_indices is None:
            print("No split set. Call set_current_fold() first.")
            return None

        summary_data = {}
        class_map_inv = {v: k for k, v in self.label_mapping.items()}

        for split_name, slide_indices in [
            ("train", self.train_slide_indices),
            ("val", self.val_slide_indices),
            ("test", self.test_slide_indices),
        ]:
            if slide_indices is None:
                continue
            labels = self.slide_data.loc[slide_indices, "label"]
            counts = labels.value_counts().sort_index()
            summary_data[split_name] = {
                class_map_inv.get(cls_idx, f"UnknownClass_{cls_idx}"): count
                for cls_idx, count in counts.items()
            }
            if self.verbose:
                print(f"\n--- {split_name.upper()} Split Summary ---")
                print(f"Number of slides: {len(slide_indices)}")
                print(
                    f"Number of patients: {len(self.slide_data.loc[slide_indices, 'case_id'].unique())}"
                )
                for cls_idx, count in counts.items():
                    class_name = class_map_inv.get(cls_idx, f"UnknownClass_{cls_idx}")
                    print(f"  Class {cls_idx} ({class_name}): {count} slides")

        if return_summary_df:
            df = pd.DataFrame(summary_data).fillna(0).astype(int)
            return df
        return None


class WSIMILDataset(Dataset):
    """
    Dataset for Multiple Instance Learning (MIL) classification tasks.

    This dataset works with both batch_size=1 and batch_size>1 when used with
    the appropriate collate function (collate_mil_features). The collate function
    ensures consistent tensor dimensions by always returning 3D tensors:
    (batch_size, max_instances, feature_dim) for features.
    """

    def __init__(
        self,
        slide_data_df: pd.DataFrame,
        data_directory: Union[str, Dict[str, str]],
        num_classes: int,  # For consistency, though label is in slide_data_df
        backbone: Optional[str] = None,
        patch_size: str = "",
        use_hdf5: bool = False,
        cache_enabled: bool = False,
        n_subsamples: int = -1,  # Number of patches to sample per bag (-1 means use all)
        site_column: Optional[str] = None,  # Column name for site ID in slide_data_df
    ):
        self.slide_data = slide_data_df  # DataFrame for this specific split
        self.data_directory = data_directory
        self.num_classes = num_classes
        self.backbone = backbone
        self.patch_size = (
            str(patch_size) if patch_size is not None else ""
        )  # Ensure string
        self.use_hdf5 = use_hdf5
        self.cache_enabled = cache_enabled
        self.n_subsamples = n_subsamples
        self.site_column = site_column
        self.data_cache: Dict[str, torch.Tensor] = {}
        self.verbose = True  # Add verbose flag if not present

        if not self.use_hdf5 and not self.backbone:
            print(
                "Warning: WSIMILDataset initialized for .pt files but backbone is not set. Call set_backbone()."
            )

    def __len__(self) -> int:
        return len(self.slide_data)

    def __getitem__(
        self, idx: int
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, np.ndarray]]:
        row = self.slide_data.iloc[idx]
        slide_id = row["slide_id"]
        slide_id = row["slide_id"]
        label = row["label"]  # Assumes 'label' is already integer mapped

        site_id = None
        if self.site_column:
            site_id = row[self.site_column]

        current_data_dir_path: str
        if isinstance(self.data_directory, dict):
            source_col = "source"  # This column must exist in slide_data_df if data_directory is a dict
            if source_col not in row.index:
                raise ValueError(
                    f"'{source_col}' column missing in slide_data for multi-source data_directory. Slide ID: {slide_id}"
                )
            source = row[source_col]
            if source not in self.data_directory:
                raise ValueError(
                    f"Source '{source}' for slide '{slide_id}' not found in data_directory keys: {list(self.data_directory.keys())}"
                )
            current_data_dir_path = self.data_directory[source]
        else:  # data_directory is a string
            current_data_dir_path = self.data_directory

        if not os.path.isdir(current_data_dir_path):
            raise FileNotFoundError(
                f"Data directory for slide {slide_id} not found: {current_data_dir_path}"
            )

        if not self.use_hdf5:
            if not self.backbone:
                raise ValueError(
                    "Backbone must be set for loading .pt files. Call set_backbone()."
                )

            # Handle cases like patch_size='512' or patch_size='' for root, vs. patch_size='256' for subdir
            patch_subdir = ""
            if (
                self.patch_size and self.patch_size != "512"
            ):  # Assuming '512' means no subdir or handled differently
                patch_subdir = self.patch_size

            # Construct path: data_dir / (optional_patch_subdir) / pt_files / backbone / slide_id.pt
            file_path = os.path.join(
                current_data_dir_path,
                patch_subdir,
                "pt_files",
                self.backbone,
                f"{slide_id}.pt",
            )

            cached_features = self.data_cache.get(file_path)
            if cached_features is not None:
                features = cached_features
            else:
                try:
                    features = torch.load(file_path)
                    if self.cache_enabled:
                        self.data_cache[file_path] = features
                except FileNotFoundError as e:
                    new_error_msg = f"Feature file not found: {file_path}. Check slide_id, backbone, patch_size, and data_directory structure."
                    print(
                        f"Details: Slide ID='{slide_id}', Backbone='{self.backbone}', Patch Size='{self.patch_size}', Dir='{current_data_dir_path}'"
                    )
                    raise FileNotFoundError(new_error_msg) from e
                except Exception as e:
                    raise RuntimeError(f"Error loading {file_path}: {e}") from e

            # Sample patches if n_subsamples is specified and bag is larger
            features = self._sample_patches(features)

            if self.site_column:
                return features, label, site_id
            return features, label
        else:  # use_hdf5
            file_path = os.path.join(current_data_dir_path, f"{slide_id}.h5")
            try:
                with _worker_hdf5_cache_lock:
                    if file_path not in _worker_hdf5_cache:
                        # Simple LRU: if cache full, pop the first item (oldest)
                        if len(_worker_hdf5_cache) > 512:
                            oldest_file = next(iter(_worker_hdf5_cache))
                            _worker_hdf5_cache[oldest_file].close()
                            del _worker_hdf5_cache[oldest_file]

                        _worker_hdf5_cache[file_path] = h5py.File(file_path, "r")

                    hdf5_file = _worker_hdf5_cache[file_path]

                features_dset = hdf5_file["features"]
                num_patches = features_dset.shape[0]

                if self.n_subsamples > 0 and num_patches > self.n_subsamples:
                    indices = torch.randperm(num_patches)[: self.n_subsamples].numpy()
                    indices.sort()
                else:
                    indices = np.arange(num_patches)
                features = torch.from_numpy(features_dset[indices])

                if "coords" in hdf5_file:
                    coords_dset = hdf5_file["coords"]
                    coordinates = coords_dset[indices]
                    if self.site_column:
                        return features, label, coordinates, site_id
                    return features, label, coordinates
                else:
                    if self.site_column:
                        return features, label, site_id
                    return features, label
            except OSError as e:
                raise OSError(f"HDF5 file not found or corrupted: {file_path}") from e

    def set_backbone(self, backbone: str) -> None:
        if self.verbose:
            print(f"Setting backbone for MILDataset: {backbone}")
        self.backbone = backbone

    def set_patch_size(self, size: Union[str, int]) -> None:
        if self.verbose:
            print(f"Setting patch size for MILDataset: {size}")
        self.patch_size = str(size)  # Ensure string

    def load_from_hdf5(self, use_hdf5: bool) -> None:
        self.use_hdf5 = use_hdf5

    def preload_data(self, num_threads: int = 8) -> None:
        if not self.cache_enabled:
            print(
                "Warning: Preloading data but cache_enabled is False. Data will not be stored in memory."
            )
            self.cache_enabled = True  # Enable it for preloading

        print(f"Preloading {len(self)} items into cache using {num_threads} threads...")
        from multiprocessing.pool import ThreadPool

        indices = list(range(len(self)))
        with ThreadPool(num_threads) as pool:
            pool.map(self.__getitem__, indices)
        print("Preloading complete.")

    def _sample_patches(self, features: torch.Tensor) -> torch.Tensor:
        """
        Sample a subset of patches from the bag if n_subsamples is specified.

        Args:
            features: Tensor of shape (num_patches, feature_dim)

        Returns:
            Sampled features tensor of shape (min(num_patches, n_subsamples), feature_dim)
        """
        if self.n_subsamples <= 0:
            return features

        num_patches = features.shape[0]
        if num_patches <= self.n_subsamples:
            return features

        # Randomly sample n_subsamples patches
        indices = torch.randperm(num_patches)[: self.n_subsamples]
        return features[indices]

    def _sample_patches_with_coords(
        self, features: torch.Tensor, coordinates: np.ndarray
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Sample a subset of patches and their corresponding coordinates together.

        Args:
            features: Tensor of shape (num_patches, feature_dim)
            coordinates: Array of shape (num_patches, ...)

        Returns:
            Tuple of (sampled_features, sampled_coordinates) with matching indices
        """
        if self.n_subsamples <= 0:
            return features, coordinates

        num_patches = features.shape[0]
        if num_patches <= self.n_subsamples:
            return features, coordinates

        # Randomly sample n_subsamples patches using same indices for both
        indices = torch.randperm(num_patches)[: self.n_subsamples].numpy()
        sampled_features = features[indices]
        sampled_coords = coordinates[indices]
        return sampled_features, sampled_coords
