import collections
import functools
import os
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,  # Added for type hinting
    RandomSampler,
    Sampler,
    SequentialSampler,
    WeightedRandomSampler,
)


def make_weights_for_balanced_classes(
    dataset: Dataset,  # Use the base torch Dataset for broader compatibility
    # Assumes dataset has:
    # 1. A way to get all labels: e.g., dataset.get_all_labels() -> List[int]
    # OR 2. Direct access to labels: e.g., dataset.labels or dataset.slide_data['label']
    # OR 3. `get_label(idx)` method as in your original (less efficient for all labels)
) -> torch.Tensor:
    """
    Creates weights for WeightedRandomSampler for handling class imbalance.
    This version is more flexible.

    Args:
        dataset: A PyTorch Dataset instance. It needs a way to access all labels.
                 Common patterns:
                 - `dataset.labels` (a list or tensor of all labels)
                 - `dataset.slide_data['label']` (if slide_data is a pandas DataFrame)
                 - A method like `dataset.get_all_labels()`

    Returns:
        torch.Tensor: A 1D tensor of weights for each sample in the dataset.
    """
    # Try to get labels in a more efficient way
    all_labels: Optional[List[int]] = None
    if hasattr(dataset, "labels") and isinstance(
        dataset.labels, (list, np.ndarray, torch.Tensor)
    ):
        all_labels = list(dataset.labels)
    elif (
        hasattr(dataset, "slide_data")
        and isinstance(dataset.slide_data, pd.DataFrame)
        and "label" in dataset.slide_data.columns
    ):
        all_labels = dataset.slide_data["label"].tolist()
    elif hasattr(dataset, "get_all_labels") and callable(dataset.get_all_labels):
        all_labels = dataset.get_all_labels()
    elif hasattr(dataset, "getlabel") and callable(
        dataset.getlabel
    ):  # Fallback to less efficient original
        print(
            "Warning: Using inefficient `getlabel(idx)` for each sample to compute weights. "
            "Consider adding a `labels` attribute or `get_all_labels()` method to your dataset for performance."
        )
        all_labels = [dataset.getlabel(i) for i in range(len(dataset))]
    else:
        raise AttributeError(
            "Dataset does not have a recognized way to access all labels "
            "(e.g., 'labels' attribute, 'slide_data[\"label\"]', "
            "or 'get_all_labels()' / 'getlabel()' method)."
        )

    if not all_labels:  # Should be caught by AttributeError, but as a safeguard
        raise ValueError("Could not extract labels from the dataset.")

    label_counts = collections.Counter(all_labels)
    num_samples = len(all_labels)

    if not label_counts:  # No labels or empty dataset
        return torch.ones(num_samples, dtype=torch.double)

    # Calculate weight for each class: N / (num_classes * count_for_that_class)
    # This is a common formula. Original: N_total_classes / count_for_that_class
    # Let's stick to a more standard approach: 1. / count_for_that_class then assign
    # Or total_samples / (num_distinct_classes * count_of_class_i)

    # Simpler: weight = 1.0 / count_of_sample's_class
    # Then WeightedRandomSampler handles it.
    # Or, if we want to give more weight to RARE classes:
    # weight_per_class = {label: num_samples / count for label, count in label_counts.items()}

    # The original implementation implied:
    # weight_per_class[class_idx] = N_total_unique_classes / num_samples_in_class_idx
    # This means dataset.slide_cls_ids was [[indices_for_class_0], [indices_for_class_1], ...]
    # Let's try to replicate that logic if `slide_cls_ids` is available, otherwise use Counter.

    if hasattr(dataset, "slide_cls_ids") and dataset.slide_cls_ids is not None:
        # This was the original logic structure
        num_distinct_classes = len(dataset.slide_cls_ids)  # N in original code
        if num_distinct_classes == 0:  # No classes defined in slide_cls_ids
            print("Warning: dataset.slide_cls_ids is empty. Returning uniform weights.")
            return torch.ones(num_samples, dtype=torch.double)

        weight_per_class_val = [0.0] * num_distinct_classes
        for i, cls_ids in enumerate(dataset.slide_cls_ids):
            if len(cls_ids) > 0:
                weight_per_class_val[i] = float(num_distinct_classes) / len(cls_ids)
            else:  # Class has no samples
                weight_per_class_val[i] = 0  # Or some other handling for empty classes

        # Check if getlabel exists before using it.
        if not hasattr(dataset, "getlabel") or not callable(dataset.getlabel):
            raise AttributeError(
                "Dataset has 'slide_cls_ids' but no 'getlabel(idx)' method "
                "to map sample index to class label for weighting."
            )

        weights = [
            weight_per_class_val[dataset.getlabel(idx)] for idx in range(num_samples)
        ]

    else:  # Fallback to using all_labels and Counter if slide_cls_ids not present
        print(
            "Warning: `dataset.slide_cls_ids` not found or is None. "
            "Calculating weights based on overall label counts."
        )
        if not label_counts:  # Already checked but good to be safe
            return torch.ones(num_samples, dtype=torch.double)

        # Standard approach: weight for a class is 1 / (number of samples in that class)
        # Then, for each sample, assign the weight of its class.
        # More robust against missing classes in label_counts if all_labels has them.
        max_label = max(all_labels) if all_labels else -1
        weight_per_class_val = [0.0] * (max_label + 1)
        for label, count in label_counts.items():
            if count > 0:
                # weight_per_class_val[label] = 1.0 / count # Basic
                weight_per_class_val[label] = num_samples / (
                    len(label_counts) * count
                )  # Another common one

        weights = [weight_per_class_val[label] for label in all_labels]

    return torch.tensor(weights, dtype=torch.double)


class SubsetSequentialSampler(Sampler[int]):  # Use Generic type hint
    """Samples elements sequentially from a given list of indices, always in the same order.

    Args:
        indices (List[int]): a sequence of indices
    """

    def __init__(self, indices: List[int]):
        if not isinstance(indices, list):
            raise TypeError("indices should be a list.")
        if not all(isinstance(i, int) for i in indices):
            raise TypeError("all elements in indices should be integers.")

        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def infer_feature_dim_from_data(
    data_directory: Union[str, dict],
    backbone: str,
    slide_ids: List[str],
    patch_size: str = "",
    use_hdf5: bool = False,
    verbose: bool = True,
) -> Optional[int]:
    """
    Infer feature dimension by loading the first available foundation-model feature file.

    Tries each slide_id until a .pt (or .h5) file is found under the expected path,
    then returns the last dimension of the feature tensor (number of features per patch).

    Path for .pt: data_directory / patch_subdir / "pt_files" / backbone / {slide_id}.pt
    Path for .h5: data_directory / {slide_id}.h5

    Args:
        data_directory: Root path(s) for feature files. If dict, tries each value.
        backbone: Backbone name (e.g. 'titan', 'resnet50') used in pt_files subdir.
        slide_ids: List of slide IDs to try (e.g. from slide_data).
        patch_size: Optional patch size subdir (e.g. '256'); empty or '512' means no subdir.
        use_hdf5: If True, look for .h5 files instead of .pt.
        verbose: If True, log when dimension is inferred.

    Returns:
        Feature dimension (int) if a file was loaded, else None.
    """
    if not slide_ids:
        return None
    # Normalize to list of (data_dir, slide_id) for uniform handling
    if isinstance(data_directory, dict):
        dirs = list(data_directory.values())
    else:
        dirs = [data_directory]
    patch_subdir = ""
    if patch_size and str(patch_size) != "512":
        patch_subdir = str(patch_size)
    for data_dir in dirs:
        if not data_dir or not os.path.isdir(data_dir):
            continue
        for slide_id in slide_ids:
            if use_hdf5:
                path = os.path.join(data_dir, f"{slide_id}.h5")
                if not os.path.isfile(path):
                    continue
                try:
                    import h5py

                    with h5py.File(path, "r") as f:
                        if "features" not in f:
                            continue
                        dim = int(f["features"].shape[-1])
                    if verbose:
                        print(f"Inferred feature dim={dim} from FM features at {path}")
                    return dim
                except Exception:
                    continue
            else:
                path = os.path.join(
                    data_dir,
                    patch_subdir,
                    "pt_files",
                    backbone,
                    f"{slide_id}.pt",
                )
                if not os.path.isfile(path):
                    continue
                try:
                    x = torch.load(path, map_location="cpu")
                    if hasattr(x, "shape"):
                        dim = int(x.shape[-1])
                    else:
                        continue
                    if verbose:
                        print(f"Inferred feature dim={dim} from FM features at {path}")
                    return dim
                except Exception:
                    continue
    return None


def collate_mil_features(
    batch: List[
        Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, torch.Tensor]]
    ],  # (features, label) or (features, label, metadata_tensor)
    n_subsamples: Optional[
        int
    ] = None,  # If provided, pad to this instead of max_instances
) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    Collate function for MIL when batch items are (Tensor_Features, Label_Int)
    or (Tensor_Features, Label_Int, Metadata_Tensor) for multi-modal.
    Handles both batch_size=1 and batch_size>1 consistently.
    Always returns 3D tensor: (batch_size, max_instances, feature_dim) for features.
    Labels are converted to a LongTensor.
    If metadata is present, returns (features, labels, metadata) with metadata (batch_size, metadata_dim).

    If n_subsamples is provided and > 0, pads all bags to n_subsamples instead of max_instances.
    This prevents excessive padding when using patch sampling.
    """
    # Filter out None items if any dataset returns None (e.g. for failed loads, though ideally handled in Dataset)
    # batch = [item for item in batch if item is not None and item[0] is not None]
    # if not batch:
    #     # Handle empty batch case, e.g. return empty tensors or raise error
    #     # This depends on how the training loop handles it.
    #     # For now, assume batch is never empty after filtering.
    #     # If it can be, the caller (DataLoader) might need a custom batch_sampler.
    #     return torch.empty(0), torch.empty(0, dtype=torch.long)

    try:
        features_list = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        has_metadata = len(batch) > 0 and len(batch[0]) >= 3

        # Handle both batch_size=1 and batch_size>1 consistently
        # Always return 3D tensor: (batch_size, max_instances, feature_dim)
        # Determine target number of instances for padding
        max_instances_in_batch = max(feat.shape[0] for feat in features_list)
        feature_dim = features_list[0].shape[1]

        # If n_subsamples is provided and > 0, use it as the padding target
        # Otherwise, use the max instances in the batch (backward compatible)
        if n_subsamples is not None and n_subsamples > 0:
            target_instances = n_subsamples
        else:
            target_instances = max_instances_in_batch

        # Pad and stack - optimized version
        # Pre-allocate output tensor for better memory efficiency
        batch_size = len(features_list)
        features = torch.zeros(
            batch_size,
            target_instances,
            feature_dim,
            dtype=features_list[0].dtype,
            device=features_list[0].device,
        )

        for i, feat in enumerate(features_list):
            num_instances = min(feat.shape[0], target_instances)
            features[i, :num_instances] = feat[:num_instances]
            # Remaining positions are already zeros from initialization

        if has_metadata:
            metadata = torch.stack([item[2] for item in batch], dim=0)
            return features, labels, metadata
    except Exception as e:
        print("Error during collation (collate_mil_features):")
        for i, item in enumerate(batch):
            print(
                f"  Item {i}: type={type(item)}, len={len(item) if isinstance(item, (tuple, list)) else 'N/A'}"
            )
            if isinstance(item, (tuple, list)) and len(item) > 0:
                print(
                    f"    Item[0] type: {type(item[0])}, shape: {item[0].shape if hasattr(item[0], 'shape') else 'N/A'}"
                )
                print(f"    Item[1] type: {type(item[1])}")
        raise RuntimeError(f"Collation failed: {e}") from e

    return features, labels


def collate_mil_survival(
    batch: List[
        Tuple[
            torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, ...]], int, float, int
        ]
    ],
    # Expected item: (path_features, omic_features, combined_label, event_time, censorship_status)
    # omic_features can be a single Tensor or a Tuple of Tensors (for coattn)
) -> Tuple[
    torch.Tensor,
    Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Collate function for Survival MIL tasks.
    Handles path features, omic features (single or tuple for coattn), and survival labels.
    Works consistently for both batch_size=1 and batch_size>1.
    Path features are concatenated (all instances from all patients in batch).
    Omic features and labels maintain batch dimension.
    """
    path_features_list = [
        item[0] for item in batch if item[0].numel() > 0
    ]  # Filter empty path features
    omic_features_list = [
        item[1] for item in batch
    ]  # Keep all, model must handle empty omics

    combined_labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    event_times = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    censorship_statuses = torch.tensor([item[4] for item in batch], dtype=torch.int)

    # Collate path features (bag of instances)
    # Concatenate all instances from all patients in the batch
    # This works consistently for both batch_size=1 and batch_size>1
    collated_path_features = (
        torch.cat(path_features_list, dim=0) if path_features_list else torch.empty(0)
    )

    # Collate omic features
    collated_omic_features: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    if isinstance(omic_features_list[0], torch.Tensor):  # All omics are single tensors
        # Stack if they are per-patient vectors, cat if they are bags of omic instances (unlikely here)
        # Assuming omic_features_list[i] is [omic_feature_dim] for patient i
        collated_omic_features = (
            torch.stack(omic_features_list, dim=0)
            if omic_features_list
            else torch.empty(0)
        )
    elif isinstance(
        omic_features_list[0], tuple
    ):  # Omic features are tuples (e.g., for CoAttn)
        # Transpose the list of tuples: [(o1a,o2a,o3a), (o1b,o2b,o3b)] -> [(o1a,o1b), (o2a,o2b), (o3a,o3b)]
        num_omic_modalities = len(omic_features_list[0])
        collated_omic_modalities = []
        for i in range(num_omic_modalities):
            modality_tensors = [
                omic_tuple[i]
                for omic_tuple in omic_features_list
                if omic_tuple[i].numel() > 0
            ]
            if modality_tensors:
                collated_omic_modalities.append(torch.stack(modality_tensors, dim=0))
            else:  # All patients had empty tensor for this modality
                collated_omic_modalities.append(torch.empty(0))
        collated_omic_features = tuple(collated_omic_modalities)
    else:
        raise TypeError(
            f"Unsupported omic feature type in batch: {type(omic_features_list[0])}"
        )

    return (
        collated_path_features,
        collated_omic_features,
        combined_labels,
        event_times,
        censorship_statuses,
    )


def get_dataloader(  # Renamed from get_split_loader for generality
    dataset: Dataset,  # Use base torch Dataset
    batch_size: int = 1,  # Default MIL batch_size is 1 (one WSI per "batch")
    shuffle: bool = False,  # For training, typically True
    use_weighted_sampler: bool = False,  # If True, enables weighted sampling for imbalance
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn_type: str = "classification",  # "classification" or "survival"
    n_subsamples: Optional[
        int
    ] = None,  # If provided, pass to collate function for padding
    # Removed `testing` flag, as subset sampling is usually for debugging/prototyping
    # and can be handled by passing a Subset of the dataset if needed.
    # device: Optional[torch.device] = None, # Not strictly needed for DataLoader creation
    persistent_workers: bool = True,
    prefetch_factor: int = 16,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for a given dataset.

    Args:
        dataset: The PyTorch Dataset instance.
        batch_size: How many samples per batch to load. For MIL, often 1.
        shuffle: Set to True to have the data reshuffled at every epoch (usually for training).
        use_weighted_sampler: If True, uses WeightedRandomSampler for class balancing.
                              Requires `make_weights_for_balanced_classes` to work with the dataset.
        num_workers: How many subprocesses to use for data loading.
        pin_memory: If True, copies Tensors into CUDA pinned memory before returning them.
        collate_fn_type: Specifies the type of collate function to use.
                         "classification" for standard (features, label) items.
                         "survival" for (path_feat, omic_feat, comb_label, time, event) items.

    Returns:
        torch.DataLoader: The configured DataLoader.
    """
    # Determine if CUDA is available and adjust defaults if necessary
    on_gpu = torch.cuda.is_available()
    current_num_workers = num_workers  # Often set to 0 for CPU for simplicity
    current_pin_memory = pin_memory if on_gpu else False

    sampler: Optional[Sampler] = None
    # Shuffle and weighted_sampler are mutually exclusive with explicitly provided sampler.
    # WeightedRandomSampler implies random sampling.
    if use_weighted_sampler:
        if shuffle is False:  # WeightedRandomSampler inherently shuffles.
            print(
                "Warning: `use_weighted_sampler=True` implies shuffling. `shuffle=False` will be ignored."
            )
        try:
            weights = make_weights_for_balanced_classes(dataset)
            # num_samples for WeightedRandomSampler is how many samples to draw in an epoch
            # Usually len(dataset) to see each sample (on average) once.
            sampler = WeightedRandomSampler(
                weights, num_samples=len(dataset), replacement=True
            )
            shuffle = False  # Sampler handles shuffling, so DataLoader's shuffle should be False
        except (AttributeError, ValueError) as e:
            print(
                f"Could not create weighted sampler: {e}. Falling back to standard RandomSampler if shuffle=True."
            )
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(
                    dataset
                )  # Should not happen if shuffle=True was intended
    elif shuffle:
        sampler = RandomSampler(dataset)
    else:  # No shuffle, no weighted sampling
        sampler = SequentialSampler(dataset)

    # Select collate_fn based on type
    # If n_subsamples is not provided, try to get it from the dataset
    if n_subsamples is None and hasattr(dataset, "n_subsamples"):
        n_subsamples = dataset.n_subsamples

    if collate_fn_type.lower() == "classification":
        # Use functools.partial to create a picklable collate function
        # This is required for Windows multiprocessing (spawn method)
        collate_function = functools.partial(
            collate_mil_features, n_subsamples=n_subsamples
        )
    elif collate_fn_type.lower() == "survival":
        collate_function = collate_mil_survival
    else:
        raise ValueError(
            f"Unknown collate_fn_type: {collate_fn_type}. "
            "Choose 'classification' or 'survival'."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_function,
        num_workers=current_num_workers,
        pin_memory=current_pin_memory,
        drop_last=False,  # Typically False for MIL unless batch_size > 1 and partial batches are an issue
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    print("DataLoader created successfully.")
    print("DataLoader parameters:")
    print(f"  batch_size: {loader.batch_size}")
    print(f"  sampler: {loader.sampler}")
    print(f"  collate_fn: {loader.collate_fn}")
    print(f"  num_workers: {loader.num_workers}")
    print(f"  pin_memory: {loader.pin_memory}")
    print("  drop_last: False")
    print("  persistent_workers: True if loader.num_workers > 0 else False")
    print("  prefetch_factor: 16 if loader.num_workers > 0 else 2")
    print(f"  dataset: {loader.dataset}")
    return loader
