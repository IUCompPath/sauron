import json
from typing import Optional

import numpy as np
import pandas as pd
import torch


class MemmapDataset(torch.utils.data.Dataset):
    """
    Dataset for Multiple Instance Learning (MIL) classification tasks using memory-mapped files.

    This dataset loads features from a binary memmap file and returns (features, label) tuples
    compatible with the standard MIL classification pipeline.
    """

    def __init__(
        self,
        bin_path: str,
        json_path: str,
        slide_data_df: Optional[pd.DataFrame] = None,
        n_subsamples: int = 2048,
    ):
        self.bin_path = bin_path
        self.n_subsamples = n_subsamples
        self.slide_data = slide_data_df

        # Load Index
        with open(json_path, "r") as f:
            self.meta = json.load(f)

        self.slides = self.meta["slides"]  # Dict: {slide_id: [start, len]}
        self.slide_ids = list(self.slides.keys())
        self.total_rows = self.meta["total_rows"]
        self.feature_dim = self.meta["feature_dim"]

        # Create a mapping from slide_id to row index in slide_data for label lookup
        if self.slide_data is not None:
            self.slide_id_to_idx = {
                row["slide_id"]: idx for idx, row in self.slide_data.iterrows()
            }
        else:
            self.slide_id_to_idx = {}

        # Placeholder for the memmap object (Lazy loading)
        # We do NOT open it here to allow multiprocessing to work safely
        self.memmap = None

    def _ensure_memmap(self):
        if self.memmap is None:
            self.memmap = np.memmap(
                self.bin_path,
                dtype="float32",
                mode="c",
                shape=(self.total_rows, self.feature_dim),
            )

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        self._ensure_memmap()

        slide_id = self.slide_ids[idx]
        start_row, num_rows = self.slides[slide_id]

        # 1. Generate Indices (0 to num_rows - 1)
        if num_rows > self.n_subsamples:
            # Random sampling
            local_indices = np.random.choice(num_rows, self.n_subsamples, replace=False)
            local_indices.sort()  # Sorting speeds up NVMe random reads
            global_indices = local_indices + start_row

            # 2. Fancy Indexing (This triggers the OS Page Cache magic)
            features = torch.from_numpy(self.memmap[global_indices])
        else:
            # Take everything if smaller than bag size
            features = torch.from_numpy(self.memmap[start_row : start_row + num_rows])

        # Get label from slide_data if available
        if self.slide_data is not None and slide_id in self.slide_id_to_idx:
            row_idx = self.slide_id_to_idx[slide_id]
            label = int(self.slide_data.iloc[row_idx]["label"])
        else:
            # Default to 0 if label not found (shouldn't happen in normal usage)
            label = 0

        return features, label


class MemmapSurvivalMILDataset(torch.utils.data.Dataset):
    """
    Dataset for Multiple Instance Learning (MIL) survival analysis tasks using memory-mapped files.

    This dataset loads features from a binary memmap file and returns survival tuples
    compatible with the standard MIL survival pipeline:
    (path_features, omic_features, combined_label, event_time, censorship_status)
    """

    def __init__(
        self,
        bin_path: str,
        json_path: str,
        patient_data_df: pd.DataFrame,
        patient_slide_dict: dict,
        time_column: str,
        event_column: str,
        disc_label_column: str,
        combined_label_column: str,
        mode: str = "path",
        n_subsamples: int = 2048,
        omic_features_df_scaled: Optional[pd.DataFrame] = None,
        omic_names_for_coattn: Optional[list] = None,
    ):
        self.bin_path = bin_path
        self.n_subsamples = n_subsamples
        self.patient_data = patient_data_df
        self.patient_slide_dict = patient_slide_dict
        self.time_col = time_column
        self.event_col = event_column
        self.disc_label_col = disc_label_column
        self.combined_label_col = combined_label_column
        self.mode = mode
        self.omic_features = omic_features_df_scaled
        self.omic_names_for_coattn = omic_names_for_coattn

        # Load Index
        with open(json_path, "r") as f:
            self.meta = json.load(f)

        self.slides = self.meta["slides"]  # Dict: {slide_id: [start, len]}
        self.total_rows = self.meta["total_rows"]
        self.feature_dim = self.meta["feature_dim"]

        # Placeholder for the memmap object (Lazy loading)
        # We do NOT open it here to allow multiprocessing to work safely
        self.memmap = None

    def _ensure_memmap(self):
        if self.memmap is None:
            self.memmap = np.memmap(
                self.bin_path,
                dtype="float32",
                mode="c",
                shape=(self.total_rows, self.feature_dim),
            )

    def __len__(self):
        return len(self.patient_data)

    def _load_path_features(self, slide_ids: list) -> torch.Tensor:
        """Load and concatenate features from multiple slides for a patient."""
        self._ensure_memmap()
        all_path_features = []

        for slide_id in slide_ids:
            if slide_id not in self.slides:
                continue  # Skip if slide not in memmap

            start_row, num_rows = self.slides[slide_id]

            # Sample patches if needed
            if num_rows > self.n_subsamples:
                local_indices = np.random.choice(
                    num_rows, self.n_subsamples, replace=False
                )
                local_indices.sort()
                global_indices = local_indices + start_row
                features = torch.from_numpy(self.memmap[global_indices])
            else:
                features = torch.from_numpy(
                    self.memmap[start_row : start_row + num_rows]
                )

            all_path_features.append(features)

        if not all_path_features:
            return torch.empty(0, self.feature_dim)

        # Concatenate all features from all slides for this patient
        combined_features = torch.cat(all_path_features, dim=0)

        # Additional sampling if total exceeds n_subsamples
        if self.n_subsamples > 0 and combined_features.shape[0] > self.n_subsamples:
            indices = torch.randperm(combined_features.shape[0])[: self.n_subsamples]
            combined_features = combined_features[indices]

        return combined_features

    def __getitem__(self, idx):
        self._ensure_memmap()

        patient_row = self.patient_data.iloc[idx]
        case_id = patient_row["case_id"]

        # Get labels and survival info
        combined_label = int(patient_row[self.combined_label_col])
        event_time = float(patient_row[self.time_col])
        censorship_status = int(patient_row[self.event_col])

        # Get slide IDs for this patient
        slide_ids = self.patient_slide_dict.get(case_id, [case_id])

        # Initialize outputs
        path_features = torch.empty(0)
        omic_data_tensor = torch.empty(0)

        # --- Load Path Features (WSI bags) ---
        if self.mode in ["path", "pathomic", "coattn", "cluster"]:
            path_features = self._load_path_features(slide_ids)

        # --- Load Omic Features ---
        if self.omic_features is not None and self.mode in [
            "omic",
            "pathomic",
            "coattn",
            "cluster",
        ]:
            if case_id in self.omic_features.index:
                if self.mode != "coattn":
                    omic_data_tensor = torch.tensor(
                        self.omic_features.loc[case_id].values, dtype=torch.float32
                    )
                else:
                    # For coattn mode, return tuple of tensors
                    if self.omic_names_for_coattn:
                        coattn_omic_tensors = []
                        for sig_omic_names in self.omic_names_for_coattn:
                            if sig_omic_names:
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
                                else:
                                    coattn_omic_tensors.append(torch.empty(0))
                            else:
                                coattn_omic_tensors.append(torch.empty(0))
                        omic_data_tensor = tuple(coattn_omic_tensors)
                    else:
                        omic_data_tensor = torch.tensor(
                            self.omic_features.loc[case_id].values, dtype=torch.float32
                        )
            else:
                # Patient not in omic_features
                if self.mode == "coattn" and self.omic_names_for_coattn:
                    omic_data_tensor = tuple(
                        torch.empty(0) for _ in self.omic_names_for_coattn
                    )
                else:
                    num_omic_features = (
                        len(self.omic_features.columns)
                        if self.omic_features is not None
                        else 0
                    )
                    omic_data_tensor = torch.zeros(
                        num_omic_features, dtype=torch.float32
                    )

        # --- Construct return tuple based on mode ---
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
            return (
                path_features,
                omic_data_tensor,
                combined_label,
                event_time,
                censorship_status,
            )
        elif self.mode == "cluster":
            cluster_ids_placeholder = torch.empty(0)
            return (
                path_features,
                cluster_ids_placeholder,
                omic_data_tensor,
                combined_label,
                event_time,
                censorship_status,
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")
