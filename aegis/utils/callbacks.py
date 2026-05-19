import os
from typing import Tuple

import numpy as np
import torch


class AccuracyLogger:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    def log(self, y_hat: int, y: int):
        self.data[y]["count"] += 1
        self.data[y]["correct"] += int(y_hat == y)

    def log_batch(self, y_hat: np.ndarray, y: np.ndarray):
        y_hat = y_hat.astype(int)
        y = y.astype(int)
        for label_class in range(self.n_classes):
            cls_mask = y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (y_hat[cls_mask] == y[cls_mask]).sum()

    def get_summary(self, c: int) -> Tuple[float, int, int]:
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]
        return (float(correct) / count if count else 0.0, correct, count)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = float("inf")
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, epoch, val_loss, model, ckpt_name="checkpoint.pt"):
        score = val_loss

        if epoch < self.warmup:
            pass
        elif score < self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience or epoch > self.stop_epoch:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, ckpt_name):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )

        # Ensure the directory exists
        os.makedirs(os.path.dirname(ckpt_name), exist_ok=True)

        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class EarlyStopping_cindex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name="checkpoint.pt"):
        score = val_loss
        # score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name: str = "checkpoint.pt"):
        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        """Saves model when validation loss decrease."""
        torch.save(model.state_dict(), ckpt_name)
