from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Tuple, Union

import h5py  # For potential H5 feature loading
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

# Import memmap dataset for conditional use
try:
    from aegis.data.memmapMILDataset import MemmapSurvivalMILDataset
except ImportError:
    MemmapSurvivalMILDataset = None

# Worker-local HDF5 file handle cache to avoid opening/closing files repeatedly
# This is shared across all dataset instances in the same worker process
_worker_hdf5_cache: Dict[str, h5py.File] = {}
_worker_hdf5_cache_lock = threading.Lock()

# Assuming generate_split and nth are from utils.utils as in original
# If these are complex, they might need to be part of this class or simplified
from aegis.utils.generic_utils import (
    generate_split,
    nth,
)


class SurvivalDataManager:
    def __init__(
        self,
        csv_path: Optional[str] = None,
        data_directory: Union[
            str, Dict[str, str]
        ] = None,  # Path to feature root dir or dict by source
        time_column: str = "survival_months",  # e.g., 'survival_months'
        event_column: str = "censorship",  # e.g., 'censorship' (0 for censored, 1 for event)
        patient_id_col_name: str = "case_id",
        slide_id_col_name: str = "slide_id",  # Though survival is often patient-level for MIL
        n_bins: int = 4,  # For discretizing survival time
        shuffle_slide_data: bool = False,  # Shuffle initial slide data loaded from CSV
        random_seed: int = 7,
        verbose: bool = True,
        patient_stratification_for_len: bool = True,  # If this manager were a dataset, how to calc len
        filter_dict: Optional[Dict[str, List[str]]] = None,
        eps: float = 1e-6,  # Epsilon for binning
        omic_csv_path: Optional[str] = None,  # Path to CSV with omic features
        omic_patient_id_col: Optional[str] = "case_id",  # Patient ID col in omic CSV
        apply_sig: bool = False,  # For coattn mode signatures
        signatures_csv_path: Optional[str] = None,  # Path to signatures CSV for coattn
        train_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv_path = test_csv
        self.data_directory = data_directory
        self.time_column = time_column
        self.event_column = event_column
        self.patient_id_col_name = patient_id_col_name
        self.slide_id_col_name = slide_id_col_name
        self.n_bins = n_bins
        self.random_seed = random_seed
        self.verbose = verbose
        self.filter_dict = filter_dict
        self.eps = eps
        self.omic_csv_path = omic_csv_path
        self.omic_patient_id_col = omic_patient_id_col
        self.apply_sig = apply_sig
        self.signatures_csv_path = signatures_csv_path

        np.random.seed(self.random_seed)

        # Load and preprocess slide/patient data
        # Load and preprocess slide/patient data
        if csv_path is not None:
            raw_data = pd.read_csv(csv_path, low_memory=False)
        elif train_csv is not None:
            try:
                train_df = pd.read_csv(train_csv, low_memory=False)
                self.train_patient_ids = (
                    train_df[self.patient_id_col_name].unique().tolist()
                )

                dfs = [train_df]
                if val_csv:
                    val_df = pd.read_csv(val_csv, low_memory=False)
                    self.val_patient_ids = (
                        val_df[self.patient_id_col_name].unique().tolist()
                    )
                    dfs.append(val_df)
                else:
                    self.val_patient_ids = []

                if test_csv:
                    test_df = pd.read_csv(test_csv, low_memory=False)
                    self.test_patient_ids = (
                        test_df[self.patient_id_col_name].unique().tolist()
                    )
                    dfs.append(test_df)
                else:
                    self.test_patient_ids = []

                raw_data = pd.concat(dfs, ignore_index=True)
            except FileNotFoundError as e:
                raise FileNotFoundError("One of the split CSV files not found") from e
        else:
            raise ValueError("Either csv_path or train_csv must be provided.")
        self.slide_data = self._rename_cols(
            raw_data
        )  # Standardizes to "case_id", "slide_id"
        self.slide_data = self._filter_data(self.slide_data, self.filter_dict)

        if shuffle_slide_data:
            self.slide_data = self.slide_data.sample(
                frac=1, random_state=self.random_seed
            ).reset_index(drop=True)

        # Patient data is primary for survival; slide_id links features to patient
        # We'll use patient_df for labels, splits, and as base for MIL dataset items
        self.patient_df = self.slide_data.drop_duplicates(subset=["case_id"]).copy()
        self.patient_df = self._discretize_survival(
            self.patient_df
        )  # Adds 'label' (combined) and 'disc_label' (bin)

        self.num_classes = len(
            self.survival_label_map
        )  # Number of (bin, event_status) combinations
        self._create_patient_slide_dictionary()  # self.patient_slide_dict

        # Omic data handling
        self.omic_features_df: Optional[pd.DataFrame] = None
        self.omic_scalers: Optional[Dict[str, StandardScaler]] = (
            None  # For different omic types if needed
        )
        self.signatures: Optional[pd.DataFrame] = None
        self.omic_names_for_coattn: Optional[List[List[str]]] = None
        self.omic_sizes_for_coattn: Optional[List[int]] = None

        if self.omic_csv_path:
            self._load_omic_data()
        if self.apply_sig and self.signatures_csv_path:
            self._load_signatures()
            if self.omic_features_df is not None:
                self._prepare_coattn_omic_names()

        self.all_patient_ids = self.patient_df["case_id"].tolist()
        self._prepare_class_indices_for_split()  # For stratified splitting using 'label'

        # Split related attributes
        self.train_patient_ids: Optional[List[str]] = None
        self.val_patient_ids: Optional[List[str]] = None
        self.test_patient_ids: Optional[List[str]] = None
        self.split_generator = None  # For k-fold from original generate_split

        if verbose:
            self._print_summary()

        # If separate CSVs were provided, we need to filter patient IDs based on available data
        if csv_path is None and train_csv is not None:
            available_patients = set(self.patient_df["case_id"].unique())
            self.train_patient_ids = [
                pid for pid in self.train_patient_ids if pid in available_patients
            ]
            self.val_patient_ids = [
                pid for pid in self.val_patient_ids if pid in available_patients
            ]
            self.test_patient_ids = [
                pid for pid in self.test_patient_ids if pid in available_patients
            ]

            # We also need to set a dummy split generator or handle it so set_next_fold doesn't break
            # Or we can just rely on the fact that train_patient_ids are set.
            # But get_mil_datasets checks for train_patient_ids, so we are good.
            # However, if the user calls set_next_fold_from_generator, it might fail or overwrite.
            # We can set a flag or dummy generator if needed.
            pass

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
                f"Patient ID column '{self.patient_id_col_name}' (expected 'case_id') not found."
            )
        # slide_id is not strictly necessary if data_directory structure uses case_id for H5/PT files
        # but good to have for mapping if multiple slides per patient contribute features.
        if "slide_id" not in df.columns:
            print(
                f"Warning: Slide ID column '{self.slide_id_col_name}' (expected 'slide_id') not found. Assuming patient-level features or direct case_id mapping for files."
            )
            # If slide_id is critical for feature loading, this might need adjustment or error.
            # For now, let's assume if multiple .pt files per patient, slide_id is used to find them.
            # If one feature file per patient (e.g. patient_id.h5), then slide_id is less critical.
            if (
                "slide_id" not in df.columns
                and self.patient_id_col_name == self.slide_id_col_name
            ):  # If slide_id is same as patient_id
                df["slide_id"] = df["case_id"]
            elif (
                "slide_id" not in df.columns
            ):  # Create a dummy slide_id if not present, can be case_id
                df["slide_id"] = df["case_id"]

        if self.time_column not in df.columns:
            raise ValueError(f"Time column '{self.time_column}' not found.")
        if self.event_column not in df.columns:
            raise ValueError(f"Event column '{self.event_column}' not found.")
        return df

    def _filter_data(
        self, data: pd.DataFrame, filter_criteria: Optional[Dict[str, List[str]]]
    ) -> pd.DataFrame:
        if filter_criteria:
            mask = pd.Series(True, index=data.index)
            for column, values in filter_criteria.items():
                if column not in data.columns:
                    print(f"Warning: Filter column '{column}' not found in data.")
                    continue
                mask &= data[column].isin(values)
            return data[mask].reset_index(drop=True)
        return data

    def _discretize_survival(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        df = patient_df.copy()

        # Patients with event (event_column == 1, assuming 0 is censored)
        uncensored_df = df[df[self.event_column] == 1]
        if (
            len(uncensored_df) < self.n_bins
        ):  # Not enough uncensored patients to form bins
            print(
                f"Warning: Only {len(uncensored_df)} uncensored patients. Reducing n_bins to {max(1, len(uncensored_df)) if len(uncensored_df) > 0 else 1}."
            )
            current_n_bins = max(1, len(uncensored_df)) if len(uncensored_df) > 0 else 1
            if current_n_bins == 0:  # No uncensored patients, create a single bin
                self.survival_bins = np.array(
                    [
                        df[self.time_column].min() - self.eps,
                        df[self.time_column].max() + self.eps,
                    ]
                )
                # assign all to bin 0
                df["disc_label"] = 0
            else:  # use qcut on uncensored for bin edges
                _, self.survival_bins = pd.qcut(
                    uncensored_df[self.time_column],
                    q=current_n_bins,
                    retbins=True,
                    labels=False,
                )
                self.survival_bins[0] = (
                    df[self.time_column].min() - self.eps
                )  # Adjust outer bins
                self.survival_bins[-1] = df[self.time_column].max() + self.eps
                df["disc_label"] = pd.cut(
                    df[self.time_column],
                    bins=self.survival_bins,
                    retbins=False,
                    labels=False,
                    right=False,
                    include_lowest=True,
                ).astype(int)

        else:  # Sufficient uncensored patients
            # Determine bins based on uncensored patients' survival times
            _, q_bins_uncensored = pd.qcut(
                uncensored_df[self.time_column],
                q=self.n_bins,
                retbins=True,
                labels=False,
            )

            # Adjust bins to cover the whole range of survival times (including censored)
            self.survival_bins = np.concatenate(
                (
                    [df[self.time_column].min() - self.eps],
                    q_bins_uncensored[1:-1],
                    [df[self.time_column].max() + self.eps],
                )
            )

            # Ensure bins are monotonically increasing (pd.qcut can sometimes produce non-monotonic due to ties)
            self.survival_bins = np.unique(self.survival_bins)
            if (
                len(self.survival_bins) < self.n_bins + 1
            ):  # If unique reduced bins too much
                print(
                    "Warning: After unique, number of bins reduced. Consider fewer n_bins or check data distribution."
                )
                # Fallback to simple linspace if qcut fails badly
                if len(self.survival_bins) <= 2:
                    self.survival_bins = np.linspace(
                        df[self.time_column].min() - self.eps,
                        df[self.time_column].max() + self.eps,
                        self.n_bins + 1,
                    )

            df["disc_label"] = pd.cut(
                df[self.time_column],
                bins=self.survival_bins,
                retbins=False,
                labels=False,
                right=False,
                include_lowest=True,
            ).astype(int)

        # Create combined label: (time_bin, event_status)
        self.survival_label_map = {}
        label_counter = 0
        # Iterate over actual unique bin numbers and event statuses
        actual_bins = sorted(df["disc_label"].unique())
        actual_event_statuses = sorted(df[self.event_column].unique())

        for bin_val in actual_bins:
            for event_val in actual_event_statuses:  # Typically 0 and 1
                self.survival_label_map[(bin_val, int(event_val))] = label_counter
                label_counter += 1

        df["label"] = df.apply(
            lambda x: self.survival_label_map[
                (x["disc_label"], int(x[self.event_column]))
            ],
            axis=1,
        )
        return df

    def _create_patient_slide_dictionary(self):
        self.patient_slide_dict = {}
        if (
            "slide_id" in self.slide_data.columns
            and "case_id" in self.slide_data.columns
        ):
            for patient_id, group in self.slide_data.groupby("case_id"):
                self.patient_slide_dict[patient_id] = group["slide_id"].tolist()
        else:  # If no slide_id, assume one "feature entity" per patient, named by case_id
            for patient_id in self.patient_df["case_id"]:
                self.patient_slide_dict[patient_id] = [
                    patient_id
                ]  # Use patient_id as the "slide_id"

    def _load_omic_data(self):
        if not self.omic_csv_path or not self.omic_patient_id_col:
            return
        try:
            omic_df = pd.read_csv(self.omic_csv_path)
            if self.omic_patient_id_col not in omic_df.columns:
                raise ValueError(
                    f"Omic patient ID column '{self.omic_patient_id_col}' not found in omic CSV."
                )

            omic_df.set_index(self.omic_patient_id_col, inplace=True)

            # Align omic data with patient_df (patients present in the main CSV)
            self.omic_features_df = omic_df.reindex(self.patient_df["case_id"]).fillna(
                0
            )  # Or other imputation

            # Identify feature columns (assuming all non-ID columns are features)
            # This might need to be more robust if omic CSV has other metadata
            self.omic_feature_names = [col for col in self.omic_features_df.columns]

            if self.verbose and self.omic_features_df is not None:
                print(
                    f"Loaded omic data for {len(self.omic_features_df)} patients, {len(self.omic_feature_names)} features."
                )
                print(
                    f"Patients in main data: {len(self.patient_df)}, Patients with omic data: {len(self.omic_features_df.dropna(how='all'))}"
                )

        except FileNotFoundError:
            print(f"Warning: Omic CSV file not found at {self.omic_csv_path}")
            self.omic_features_df = None
        except Exception as e:
            print(f"Error loading omic data: {e}")
            self.omic_features_df = None

    def _load_signatures(self):
        if not self.signatures_csv_path:
            return
        try:
            self.signatures = pd.read_csv(self.signatures_csv_path)
            if self.verbose:
                print(f"Loaded signatures from {self.signatures_csv_path}")
        except FileNotFoundError:
            print(f"Warning: Signatures CSV not found at {self.signatures_csv_path}")
            self.signatures = None

    def _prepare_coattn_omic_names(self):
        if self.signatures is None or self.omic_features_df is None:
            return

        self.omic_names_for_coattn = []
        # Original code implies signatures CSV has columns, each being a gene set / signature name
        # And values are gene names, possibly with suffixes like _mut, _cnv, _rnaseq
        available_omic_cols = pd.Series(self.omic_features_df.columns)

        for sig_col_name in self.signatures.columns:
            # Genes in the current signature
            genes_in_sig = self.signatures[sig_col_name].dropna().unique()

            # Construct potential feature names (gene + suffix)
            potential_features = []
            for gene in genes_in_sig:
                for suffix in [
                    "_mut",
                    "_cnv",
                    "_rnaseq",
                    "",
                ]:  # Add empty for base gene name
                    potential_features.append(f"{gene}{suffix}")

            # Find which of these potential features actually exist in our omic_features_df
            intersecting_features = sorted(
                list(set(potential_features) & set(available_omic_cols))
            )
            self.omic_names_for_coattn.append(intersecting_features)

        self.omic_sizes_for_coattn = [
            len(names) for names in self.omic_names_for_coattn
        ]
        if self.verbose:
            print("Prepared omic names for CoAttn:")
            for i, names in enumerate(self.omic_names_for_coattn):
                print(f"  Signature {i}: {len(names)} features")

    def _prepare_class_indices_for_split(self) -> None:
        # For stratified splitting based on the combined (bin, event) label
        self.patient_class_labels_for_stratification = self.patient_df["label"].values
        self.num_unique_stratification_labels = len(
            np.unique(self.patient_class_labels_for_stratification)
        )

        self.patient_indices_by_strat_label = [
            np.where(self.patient_class_labels_for_stratification == strat_label)[0]
            for strat_label in range(
                self.num_unique_stratification_labels
            )  # Assumes labels are 0 to N-1
        ]

    def _print_summary(self) -> None:
        print("--- Survival DataManager Summary ---")
        print(f"CSV Path: {self.csv_path}")
        print(f"Time Column: {self.time_column}, Event Column: {self.event_column}")
        print(f"N Bins for Discretization: {self.n_bins}")
        print(f"Actual Survival Bins Edges: {self.survival_bins}")
        print(f"Survival Label Map (bin, event) -> int: {self.survival_label_map}")
        print(f"Number of Combined Survival Classes: {self.num_classes}")
        print(f"Total unique patients in manager: {len(self.patient_df)}")
        print("Patient-level combined label counts:")
        print(self.patient_df["label"].value_counts(sort=False))
        if self.omic_features_df is not None:
            print(
                f"Omic features loaded for {self.omic_features_df.shape[0]} patients, with {self.omic_features_df.shape[1]} features."
            )
        if self.signatures is not None:
            print(
                f"Signatures loaded. {len(self.omic_names_for_coattn or [])} sets for CoAttn."
            )
        print("------------------------------------")

    def create_splits_from_generating_function(
        self,
        k=3,
        val_num=(25, 25),
        test_num=(40, 40),
        label_frac=1.0,
        custom_test_ids=None,
    ):
        if self.train_patient_ids is not None:
            if self.verbose:
                print("Splits already set from separate CSVs. Using fixed split.")

            # Find indices of current split IDs in patient_df
            train_indices = self.patient_df.index[
                self.patient_df["case_id"].isin(self.train_patient_ids)
            ].tolist()
            val_indices = self.patient_df.index[
                self.patient_df["case_id"].isin(self.val_patient_ids)
            ].tolist()
            test_indices = self.patient_df.index[
                self.patient_df["case_id"].isin(self.test_patient_ids)
            ].tolist()

            def dummy_gen():
                yield (train_indices, val_indices, test_indices)

            self.split_generator = dummy_gen()
            self.num_folds_generated = 1
            return

        if not hasattr(
            self, "patient_indices_by_strat_label"
        ):  # Ensure _prepare_class_indices_for_split was called
            self._prepare_class_indices_for_split()

        settings = {
            "n_splits": k,
            "val_num": val_num,  # These numbers might be absolute counts or % depending on generate_split
            "test_num": test_num,
            "label_frac": label_frac,
            "seed": self.random_seed,
            "custom_test_ids": custom_test_ids,  # This needs careful handling if IDs are strings
            "cls_ids": self.patient_indices_by_strat_label,  # Indices into patient_df
            "samples": len(self.patient_df),  # Total number of patients
        }
        self.split_generator = generate_split(**settings)
        self.num_folds_generated = k  # Assuming k folds are generated
        if self.verbose:
            print(f"Initialized split generator for {k} folds.")

    def set_next_fold_from_generator(
        self, start_from_fold: Optional[int] = None
    ) -> bool:
        if self.split_generator is None:
            raise RuntimeError(
                "Split generator not initialized. Call create_splits_from_generating_function() first."
            )

        try:
            if start_from_fold is not None:  # Fast-forward generator
                # Ensure generator is reset or handled if called multiple times with start_from_fold
                # This might require re-initializing the generator if it's stateful and consumed.
                # For now, assume nth can be used if the generator is fresh or resettable.
                split_indices_in_patient_df = nth(self.split_generator, start_from_fold)
            else:
                split_indices_in_patient_df = next(
                    self.split_generator
                )  # Advances the generator
        except StopIteration:
            if self.verbose:
                print("No more folds in the generator.")
            return False  # No more folds

        train_indices, val_indices, test_indices = split_indices_in_patient_df

        self.train_patient_ids = self.patient_df.iloc[train_indices]["case_id"].tolist()
        self.val_patient_ids = self.patient_df.iloc[val_indices]["case_id"].tolist()
        self.test_patient_ids = self.patient_df.iloc[test_indices]["case_id"].tolist()

        if self.verbose:
            print(
                f"Set fold: Train Pat. {len(self.train_patient_ids)}, Val Pat. {len(self.val_patient_ids)}, Test Pat. {len(self.test_patient_ids)}"
            )

        # Normalize omic data if present, fitting on current TRAIN split
        if self.omic_features_df is not None and self.train_patient_ids:
            self._fit_omic_scaler(self.train_patient_ids)
        return True

    def get_mil_datasets(
        self,
        mode: str,  # 'path', 'omic', 'pathomic', 'coattn'
        use_hdf5: bool = False,  # For path features
        backbone: Optional[str] = None,  # For path .pt features
        patch_size: str = "",  # For path .pt features
        cache_enabled: bool = False,
        n_subsamples: int = -1,
        memmap_bin_path: Optional[str] = None,  # Path to memmap binary file
        memmap_json_path: Optional[str] = None,  # Path to memmap index JSON file
    ) -> Tuple[
        Optional[SurvivalMILDataset],
        Optional[SurvivalMILDataset],
        Optional[SurvivalMILDataset],
    ]:
        if self.train_patient_ids is None:  # Check if a fold has been set
            raise ValueError(
                "Splits not set. Call a split creation method and set_next_fold...() first."
            )

        # Check if memmap paths are provided
        use_memmap = memmap_bin_path is not None and memmap_json_path is not None

        if use_memmap:
            # Use MemmapSurvivalMILDataset
            if MemmapSurvivalMILDataset is None:
                raise ImportError(
                    "MemmapSurvivalMILDataset not available. "
                    "Ensure aegis.data.memmapMILDataset is properly imported."
                )

            datasets = []
            for split_name, patient_ids_list in [
                ("train", self.train_patient_ids),
                ("val", self.val_patient_ids),
                ("test", self.test_patient_ids),
            ]:
                if not patient_ids_list:
                    datasets.append(None)
                    continue

                # Get patient data for this split
                current_split_patient_df = self.patient_df[
                    self.patient_df["case_id"].isin(patient_ids_list)
                ].reset_index(drop=True)

                current_split_omic_df = None
                if self.omic_features_df is not None:
                    # Select omic features for current patients
                    omic_for_split = self.omic_features_df.loc[patient_ids_list]
                    # Apply scaler if fitted (scaler is fitted on train_patient_ids)
                    if (
                        self.omic_scalers and "all" in self.omic_scalers
                    ):  # Assuming one scaler for all omics for now
                        scaled_omic_data = self.omic_scalers["all"].transform(
                            omic_for_split[self.omic_feature_names]
                        )
                        current_split_omic_df = pd.DataFrame(
                            scaled_omic_data,
                            columns=self.omic_feature_names,
                            index=omic_for_split.index,
                        )
                    else:
                        current_split_omic_df = omic_for_split[
                            self.omic_feature_names
                        ].copy()  # No scaling or scaler not ready

                dataset = MemmapSurvivalMILDataset(
                    bin_path=memmap_bin_path,
                    json_path=memmap_json_path,
                    patient_data_df=current_split_patient_df,
                    patient_slide_dict=self.patient_slide_dict,
                    time_column=self.time_column,
                    event_column=self.event_column,
                    disc_label_column="disc_label",
                    combined_label_column="label",
                    mode=mode,
                    n_subsamples=n_subsamples if n_subsamples > 0 else 2048,
                    omic_features_df_scaled=current_split_omic_df,
                    omic_names_for_coattn=self.omic_names_for_coattn,
                )
                datasets.append(dataset)

            return tuple(datasets)
        else:
            # Use original SurvivalMILDataset
            common_params = {
                "patient_slide_dict": self.patient_slide_dict,
                "data_directory": self.data_directory,
                "time_column": self.time_column,
                "event_column": self.event_column,
                "disc_label_column": "disc_label",  # Column name in patient_df for discrete time bin
                "combined_label_column": "label",  # Column name in patient_df for (bin,event) label
                "mode": mode,
                "use_hdf5": use_hdf5,
                "backbone": backbone,
                "patch_size": patch_size,
                "cache_enabled": cache_enabled,
                "n_subsamples": n_subsamples,
                "omic_names_for_coattn": self.omic_names_for_coattn,  # Pass coattn specific omic names
            }

            datasets = []
            for split_name, patient_ids_list in [
                ("train", self.train_patient_ids),
                ("val", self.val_patient_ids),
                ("test", self.test_patient_ids),
            ]:
                if not patient_ids_list:
                    datasets.append(None)
                    continue

                # Get patient data for this split
                current_split_patient_df = self.patient_df[
                    self.patient_df["case_id"].isin(patient_ids_list)
                ].reset_index(drop=True)

                current_split_omic_df = None
                if self.omic_features_df is not None:
                    # Select omic features for current patients
                    omic_for_split = self.omic_features_df.loc[patient_ids_list]
                    # Apply scaler if fitted (scaler is fitted on train_patient_ids)
                    if (
                        self.omic_scalers and "all" in self.omic_scalers
                    ):  # Assuming one scaler for all omics for now
                        scaled_omic_data = self.omic_scalers["all"].transform(
                            omic_for_split[self.omic_feature_names]
                        )
                        current_split_omic_df = pd.DataFrame(
                            scaled_omic_data,
                            columns=self.omic_feature_names,
                            index=omic_for_split.index,
                        )
                    else:
                        current_split_omic_df = omic_for_split[
                            self.omic_feature_names
                        ].copy()  # No scaling or scaler not ready

                dataset = SurvivalMILDataset(
                    patient_data_df=current_split_patient_df,
                    omic_features_df_scaled=current_split_omic_df,  # Pass potentially scaled omics
                    **common_params,
                )
                datasets.append(dataset)

            return tuple(datasets)

    def _fit_omic_scaler(self, train_patient_ids_for_scaling: List[str]):
        if self.omic_features_df is None or not train_patient_ids_for_scaling:
            self.omic_scalers = None
            return

        # Ensure all train_patient_ids are in omic_features_df index
        train_patient_ids_for_scaling = [
            pid
            for pid in train_patient_ids_for_scaling
            if pid in self.omic_features_df.index
        ]
        if not train_patient_ids_for_scaling:
            print(
                "Warning: No training patients found in omic_features_df for scaling."
            )
            self.omic_scalers = None
            return

        train_omic_data = self.omic_features_df.loc[
            train_patient_ids_for_scaling, self.omic_feature_names
        ]

        # For now, one scaler for all omic features. Can be extended.
        scaler = StandardScaler()
        scaler.fit(train_omic_data)
        self.omic_scalers = {"all": scaler}
        if self.verbose:
            print(
                "Fitted StandardScaler for omic features on the current training split."
            )

    def save_current_split_patient_ids(self, filename: str):
        if self.train_patient_ids is None:
            raise ValueError("Splits not set.")
        max_len = max(
            len(self.train_patient_ids or []),
            len(self.val_patient_ids or []),
            len(self.test_patient_ids or []),
        )

        df = pd.DataFrame(
            {
                "train_ids": pd.Series(self.train_patient_ids or []).reindex(
                    range(max_len)
                ),
                "val_ids": pd.Series(self.val_patient_ids or []).reindex(
                    range(max_len)
                ),
                "test_ids": pd.Series(self.test_patient_ids or []).reindex(
                    range(max_len)
                ),
            }
        )
        df.to_csv(filename, index=False)
        if self.verbose:
            print(f"Survival split patient IDs saved to {filename}")


class SurvivalMILDataset(Dataset):
    """
    Dataset for Multiple Instance Learning (MIL) survival analysis tasks.
    """

    def __init__(
        self,
        patient_data_df: pd.DataFrame,  # DF for patients in this specific split
        patient_slide_dict: Dict[
            str, List[str]
        ],  # Full {patient_id: [slide_ids]} mapping
        data_directory: Union[str, Dict[str, str]],
        time_column: str,
        event_column: str,
        disc_label_column: str,  # e.g. "disc_label"
        combined_label_column: str,  # e.g. "label"
        mode: str,  # 'path', 'omic', 'pathomic', 'coattn'
        use_hdf5: bool = False,
        backbone: Optional[str] = None,
        patch_size: str = "",
        cache_enabled: bool = False,
        n_subsamples: int = -1,  # Number of patches to sample per bag (-1 means use all)
        omic_features_df_scaled: Optional[
            pd.DataFrame
        ] = None,  # Scaled omic features for this split
        omic_names_for_coattn: Optional[List[List[str]]] = None,
    ):
        self.patient_data = patient_data_df
        self.patient_slide_dict = patient_slide_dict
        self.data_directory = data_directory
        self.time_col = time_column
        self.event_col = event_column
        self.disc_label_col = disc_label_column
        self.combined_label_col = combined_label_column
        self.mode = mode
        self.use_hdf5 = use_hdf5  # For path features
        self.backbone = backbone  # For path .pt features
        self.patch_size = str(patch_size) if patch_size is not None else ""
        self.cache_enabled = cache_enabled
        self.n_subsamples = n_subsamples
        self.path_features_cache: Dict[str, torch.Tensor] = (
            {}
        )  # Cache for loaded slide features

        self.omic_features = (
            omic_features_df_scaled  # Already selected and scaled for this split
        )
        self.omic_names_for_coattn = omic_names_for_coattn

        valid_modes = [
            "path",
            "omic",
            "pathomic",
            "coattn",
            "cluster",
        ]  # Added cluster from original
        if self.mode not in valid_modes:
            raise ValueError(
                f"Mode '{self.mode}' not implemented. Valid modes: {valid_modes}"
            )

        if "omic" in self.mode and self.omic_features is None:
            print(
                f"Warning: Mode '{self.mode}' requires omic features, but none were provided/loaded."
            )
        if self.mode == "coattn" and not self.omic_names_for_coattn:
            print(
                "Warning: Mode 'coattn' selected, but omic_names_for_coattn not provided."
            )
        if "path" in self.mode and not self.use_hdf5 and not self.backbone:
            print("Warning: Path-based mode with .pt files, but backbone is not set.")

    def __len__(self) -> int:
        return len(self.patient_data)

    def _load_path_features(
        self, patient_case_id: str, slide_ids_for_patient: List[str]
    ) -> torch.Tensor:
        all_path_features = []
        for slide_id in slide_ids_for_patient:
            # Determine data_dir for this slide (if self.data_directory is a dict)
            current_data_dir_path = self.data_directory
            if isinstance(self.data_directory, dict):
                # This requires 'source' column in the main slide_data from which patient_slide_dict was built
                # For simplicity, assume if data_directory is dict, it applies globally or needs more complex handling
                # Or, assume patient_data_df has a 'source' column if data_directory is a dict
                row_for_source = (
                    self.patient_data[
                        self.patient_data["case_id"] == patient_case_id
                    ].iloc[0]
                    if "source" in self.patient_data.columns
                    else None
                )  # Hacky
                if row_for_source is not None and "source" in row_for_source.index:
                    source = row_for_source["source"]
                    if source not in self.data_directory:
                        raise ValueError(
                            f"Source '{source}' for patient '{patient_case_id}' not in data_directory keys."
                        )
                    current_data_dir_path = self.data_directory[source]
                # else:
                # If no source per patient, and data_directory is dict, this is an issue.
                # Default to first key or raise error if not resolvable.
                # For now, assume string or resolvable dict.

            if not self.use_hdf5:
                if not self.backbone:
                    raise ValueError("Backbone needed for .pt files.")
                patch_subdir = ""
                if self.patch_size and self.patch_size != "512":
                    patch_subdir = self.patch_size
                file_path = os.path.join(
                    current_data_dir_path,
                    patch_subdir,
                    "pt_files",
                    self.backbone,
                    f"{slide_id}.pt",
                )

                if file_path in self.path_features_cache:
                    wsi_bag = self.path_features_cache[file_path]
                else:
                    try:
                        wsi_bag = torch.load(file_path)
                        if self.cache_enabled:
                            self.path_features_cache[file_path] = wsi_bag
                    except FileNotFoundError:
                        # Try without patch_subdir for robustness if original was inconsistent
                        file_path_alt = os.path.join(
                            current_data_dir_path,
                            "pt_files",
                            self.backbone,
                            f"{slide_id}.pt",
                        )
                        try:
                            wsi_bag = torch.load(file_path_alt)
                            if self.cache_enabled:
                                self.path_features_cache[file_path_alt] = wsi_bag
                        except FileNotFoundError:
                            raise FileNotFoundError(
                                f"Path feature file not found for slide {slide_id} at {file_path} or {file_path_alt}"
                            )
                all_path_features.append(wsi_bag)
            else:  # use_hdf5
                h5_file_path = os.path.join(
                    current_data_dir_path, "h5_files", f"{slide_id}.h5"
                )
                try:
                    with _worker_hdf5_cache_lock:
                        if h5_file_path not in _worker_hdf5_cache:
                            _worker_hdf5_cache[h5_file_path] = h5py.File(
                                h5_file_path, "r"
                            )
                        hf = _worker_hdf5_cache[h5_file_path]
                    features = torch.from_numpy(hf["features"][:])
                    all_path_features.append(features)
                except OSError:
                    raise OSError(
                        f"HDF5 file not found or corrupted for slide {slide_id} at {h5_file_path}"
                    )

        if not all_path_features:  # No features found for any slide_id of this patient
            # Return a dummy tensor or raise error. For MIL, usually expect some features.
            # Shape needs to match what model expects if this patient has no path data.
            # This depends on feature dim, e.g., (0, 1024)
            print(
                f"Warning: No path features loaded for patient {patient_case_id} (slides: {slide_ids_for_patient}). Returning zero tensor."
            )
            # Try to infer feature dim from backbone or a fixed value
            # This is tricky. For now, a small placeholder. Model must handle 0-dim input.
            return torch.zeros(
                (0, 1)
            )  # Or try to get feature dim if one file was loaded before

        if not all_path_features:
            return torch.zeros((0, 1))

        # Concatenate all features from all slides for this patient
        combined_features = torch.cat(all_path_features, dim=0)

        # Sample patches if n_subsamples is specified and bag is larger
        if self.n_subsamples > 0:
            num_patches = combined_features.shape[0]
            if num_patches > self.n_subsamples:
                indices = torch.randperm(num_patches)[: self.n_subsamples]
                combined_features = combined_features[indices]

        return combined_features

    def __getitem__(self, idx: int) -> tuple:
        patient_row = self.patient_data.iloc[idx]
        case_id = patient_row["case_id"]

        # Labels and time/event info
        # discrete_time_bin = patient_row[self.disc_label_col] # Bin index
        combined_label = patient_row[
            self.combined_label_col
        ]  # (Bin, Event) mapped to int
        event_time = patient_row[self.time_col]
        censorship_status = patient_row[self.event_col]  # 0=censored, 1=event

        slide_ids = self.patient_slide_dict.get(
            case_id, [case_id]
        )  # Fallback to case_id if not in dict

        # Initialize outputs
        path_features = torch.empty(0)  # Placeholder
        omic_data_tensor = torch.empty(0)

        # --- Load Path Features (WSI bags) ---
        if self.mode in ["path", "pathomic", "coattn", "cluster"]:
            path_features = self._load_path_features(case_id, slide_ids)

        # --- Load Omic Features ---
        if self.omic_features is not None and self.mode in [
            "omic",
            "pathomic",
            "coattn",
            "cluster",
        ]:
            if case_id in self.omic_features.index:
                # For 'omic', 'pathomic', 'cluster' - all omics for the patient
                if self.mode != "coattn":
                    omic_data_tensor = torch.tensor(
                        self.omic_features.loc[case_id].values, dtype=torch.float32
                    )
                # For 'coattn' - specific sets of omics
                else:
                    if not self.omic_names_for_coattn:
                        # Return all omics if coattn names not set, or error, or empty
                        print(
                            f"Warning: CoAttn mode but no omic_names_for_coattn for patient {case_id}. Using all omics or empty."
                        )
                        omic_data_tensor = (
                            torch.tensor(
                                self.omic_features.loc[case_id].values,
                                dtype=torch.float32,
                            )
                            if not self.omic_features.empty
                            else torch.empty(0)
                        )

                        # Or, if CoAttn expects multiple omic tensors:
                        # For CoAttn, original returned multiple omic tensors.
                        # This structure needs to be decided. For now, one tensor or a list.
                        # To match original:
                        coattn_omic_tensors = []
                        if self.omic_names_for_coattn:
                            for i, sig_omic_names in enumerate(
                                self.omic_names_for_coattn
                            ):
                                if (
                                    sig_omic_names
                                ):  # If this signature has features defined
                                    # Ensure all names are in omic_features columns
                                    valid_sig_omic_names = [
                                        name
                                        for name in sig_omic_names
                                        if name in self.omic_features.columns
                                    ]
                                    if valid_sig_omic_names:
                                        coattn_omic_tensors.append(
                                            torch.tensor(
                                                self.omic_features.loc[
                                                    case_id, valid_sig_omic_names
                                                ].values,
                                                dtype=torch.float32,
                                            )
                                        )
                                    else:  # No valid features for this signature for this patient
                                        coattn_omic_tensors.append(
                                            torch.empty(0)
                                        )  # Placeholder for this signature
                                else:  # No features defined for this signature
                                    coattn_omic_tensors.append(torch.empty(0))
                        # This changes the return signature for __getitem__ based on mode.
                        # PyTorch DataLoader usually expects consistent return types.
                        # It's better to return a dict or a fixed tuple structure.
                        # For now, let's stick to path_features, omic_data_tensor, label, event_time, c
                        # And CoAttn model would need to split omic_data_tensor if it's concatenated.
                        # OR, if mode is 'coattn', omic_data_tensor could be a list/tuple of tensors.
                        # Let's assume for coattn, omic_data_tensor IS the list of tensors for now.
                        if self.mode == "coattn":
                            omic_data_tensor = (
                                tuple(coattn_omic_tensors)
                                if coattn_omic_tensors
                                else tuple(
                                    torch.empty(0)
                                    for _ in range(
                                        len(self.omic_names_for_coattn or [])
                                    )
                                )
                            )

            else:  # Patient not in omic_features (e.g. missing data)
                print(
                    f"Warning: Omic data not found for patient {case_id}. Returning empty tensor for omics."
                )
                if self.mode == "coattn" and self.omic_names_for_coattn:
                    omic_data_tensor = tuple(
                        torch.empty(0) for _ in self.omic_names_for_coattn
                    )
                else:  # For other omic modes
                    # Try to get expected shape for omic from the dataframe columns
                    num_omic_features = (
                        len(self.omic_features.columns)
                        if self.omic_features is not None
                        else 0
                    )
                    omic_data_tensor = torch.zeros(
                        num_omic_features, dtype=torch.float32
                    )

        # --- Construct return tuple based on mode ---
        # This needs to be consistent for the DataLoader.
        # (path_features, omic_features, label (combined), event_time, censorship_status)
        # If a mode doesn't use one, it can be a placeholder (e.g., torch.empty(0)).

        if self.mode == "path":
            return (
                path_features,
                torch.empty(0),
                combined_label,
                event_time,
                censorship_status,
            )
        elif self.mode == "omic":
            return (
                torch.empty(0),
                omic_data_tensor,
                combined_label,
                event_time,
                censorship_status,
            )
        elif self.mode == "pathomic":
            return (
                path_features,
                omic_data_tensor,
                combined_label,
                event_time,
                censorship_status,
            )
        elif self.mode == "coattn":
            # `omic_data_tensor` is already a tuple of tensors for coattn here
            return (
                path_features,
                omic_data_tensor,
                combined_label,
                event_time,
                censorship_status,
            )
        elif self.mode == "cluster":  # Original 'cluster' mode also had 'cluster_ids'
            # cluster_ids logic was: self.fname2ids[slide_id[:-4]+'.pt']
            # This fname2ids needs to be loaded and passed, typically by SurvivalDataManager
            # For now, returning placeholder for cluster_ids
            cluster_ids_placeholder = torch.empty(0)  # Placeholder
            return (
                path_features,
                cluster_ids_placeholder,
                omic_data_tensor,
                combined_label,
                event_time,
                censorship_status,
            )
        else:  # Should not happen due to check in __init__
            raise NotImplementedError(f"Mode {self.mode} data assembly not defined.")

    def set_backbone(self, backbone: str):
        self.backbone = backbone

    def set_patch_size(self, size: str):
        self.patch_size = str(size)

    def load_from_hdf5(self, use_hdf5: bool):
        self.use_hdf5 = use_hdf5
