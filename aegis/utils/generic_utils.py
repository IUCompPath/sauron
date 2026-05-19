import argparse
import collections
import math
import os
import pickle
from itertools import islice
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


def save_pkl(filename: str, save_object: object) -> None:
    with open(filename, "wb") as f:
        pickle.dump(save_object, f)


def load_pkl(filename: str) -> object:
    with open(filename, "rb") as file:
        return pickle.load(file)


def seed_everything(seed: int = 42) -> None:
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_error(Y_hat: torch.Tensor, Y: torch.Tensor) -> float:
    return 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()


def save_splits(
    split_datasets: List[Dataset],
    column_keys: List[str],
    filename: str,
    boolean_style: bool = False,
) -> None:
    splits = [
        split_datasets[i].slide_data["slide_id"] for i in range(len(split_datasets))
    ]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=["train", "val", "test"])

    df.to_csv(filename)
    print(f"Splits saved to {filename}")


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def log_results(
    df: pd.DataFrame, args: argparse.Namespace, writer: SummaryWriter
) -> None:
    mean_metrics = df.mean()
    std_metrics = df.std()

    for metric in ["test_auc", "val_auc", "test_acc", "val_acc"]:
        writer.add_scalar(f"mean_{metric}", mean_metrics[metric], args.k_end)
        writer.add_scalar(f"std_{metric}", std_metrics[metric], args.k_end)

    df_append = pd.DataFrame(
        {
            "folds": ["mean", "std"],
            **{
                metric: [mean_metrics[metric], std_metrics[metric]]
                for metric in ["test_auc", "val_auc", "test_acc", "val_acc"]
            },
        }
    )

    final_df = pd.concat([df, df_append])
    save_name = (
        f"summary_partial_{args.k_start}_{args.k_end}.csv"
        if args.k_end - args.k_start != args.k
        else "summary.csv"
    )
    final_df.to_csv(os.path.join(args.results_dir, save_name), index=False)


def generate_split(
    cls_ids,
    val_num,
    test_num,
    samples,
    n_splits=5,
    seed=7,
    label_frac=1.0,
    custom_test_ids=None,
):
    """
    Generates train, validation, and test splits from a dataset.

    This is a generator function that yields indices for each fold. It supports
    stratified splitting to ensure that class distributions are maintained across splits.

    Args:
        cls_ids (list of np.array): A list where each element is an array of indices
                                    belonging to a specific class.
        val_num (tuple/list): The number of samples from each class to include in the
                              validation set.
        test_num (tuple/list): The number of samples from each class to include in the
                               test set.
        samples (int): The total number of samples in the dataset.
        n_splits (int): The number of folds to generate.
        seed (int): The random seed for reproducibility.
        label_frac (float): The fraction of training labels to use.
        custom_test_ids (list, optional): A pre-defined list of test indices to use.

    Yields:
        tuple: A tuple containing (train_ids, val_ids, test_ids) for a fold.
    """
    indices = np.arange(samples)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)
            remaining_ids = possible_indices

            if val_num[c] > 0 and len(possible_indices) > val_num[c]:
                val_ids = np.random.choice(possible_indices, val_num[c], replace=False)
                remaining_ids = np.setdiff1d(possible_indices, val_ids)
                all_val_ids.extend(val_ids)

            if (
                custom_test_ids is None
                and test_num[c] > 0
                and len(remaining_ids) > test_num[c]
            ):
                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1.0:
                sampled_train_ids.extend(remaining_ids)
            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sorted(sampled_train_ids), sorted(all_val_ids), sorted(all_test_ids)


def nth(iterator, n, default=None):
    """
    Returns the n-th item or a default value from an iterator.
    This is used to jump to a specific fold in the split generator without
    iterating through all the previous ones.

    Args:
        iterator: The iterator to consume.
        n (int): The index of the item to retrieve.
        default: The value to return if the iterator is exhausted.

    Returns:
        The n-th item from the iterator or the default value.
    """
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)
