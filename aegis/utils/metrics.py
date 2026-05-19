
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sksurv.metrics import concordance_index_censored


# --- Metric Calculation Helpers ---
def _calculate_classification_auc(
    all_labels_np: np.ndarray, all_probs_np: np.ndarray, n_classes: int
) -> float:
    """Calculates AUC for classification."""
    if n_classes == 2:
        # Ensure there are at least two classes in labels for AUC calculation
        if len(np.unique(all_labels_np)) < 2:
            print(
                "Warning: Only one class present in labels for binary AUC calculation. AUC set to 0.0."
            )
            return 0.0
        try:
            return roc_auc_score(all_labels_np, all_probs_np[:, 1])
        except ValueError as e:
            print(
                f"Warning: Could not calculate AUC (binary): {e}. Check if all_probs_np has two columns."
            )
            return 0.0
    else:  # Multi-class
        try:
            # Binarize labels against all potential classes (0 to n_classes-1)
            # This ensures consistent shape for roc_auc_score's `average` parameter.
            all_labels_bin = label_binarize(all_labels_np, classes=range(n_classes))

            # Check if any class has no instances after binarization.
            # Some classes might not be present in this specific batch/dataset split.
            # roc_auc_score (ovr) can handle cases where some classes specified in `labels`
            # are not present, as long as y_true and y_score shapes match on the number of classes.
            # However, if all_labels_bin has fewer columns than n_classes (e.g. max label is 1 for n_classes=3)
            # then roc_auc_score needs y_score to match that shape or careful slicing.
            # Assuming all_probs_np always has n_classes columns.
            if all_labels_bin.shape[1] < n_classes:
                # This can happen if the max label in all_labels_np is less than n_classes-1.
                # Pad all_labels_bin with zero columns for missing classes up to n_classes.
                padding = np.zeros(
                    (all_labels_bin.shape[0], n_classes - all_labels_bin.shape[1])
                )
                all_labels_bin = np.hstack((all_labels_bin, padding))

            # Ensure at least two classes are present in the actual data for meaningful OvR AUC
            if len(np.unique(all_labels_np)) < 2:
                print(
                    "Warning: Less than 2 unique classes present in labels for multi-class AUC. AUC set to 0.0."
                )
                return 0.0

            return roc_auc_score(
                all_labels_bin, all_probs_np, multi_class="ovr", average="weighted"
            )
        except ValueError as e:
            # This can happen if all_probs_np doesn't have n_classes columns, or other inconsistencies.
            print(f"Warning: Could not calculate AUC (multi-class, weighted OvR): {e}")
            return 0.0


def _calculate_survival_c_index(
    all_event_times_np: np.ndarray,
    all_censorships_np: np.ndarray,
    all_risks_np: np.ndarray,
) -> float:
    """Calculates C-Index for survival."""
    if (
        len(all_event_times_np) == 0
        or len(all_censorships_np) == 0
        or len(all_risks_np) == 0
    ):
        print("Warning: Empty arrays provided for C-index calculation. Returning 0.0.")
        return 0.0
    event_observed = (1 - all_censorships_np).astype(bool)
    try:
        # Check for trivial cases that concordance_index_censored might not handle well
        if (
            len(np.unique(event_observed)) == 1 and not event_observed[0]
        ):  # All censored
            print(
                "Warning: All samples are censored. C-index is undefined, returning 0.0."
            )
            return 0.0
        if len(np.unique(all_event_times_np[event_observed])) < 1 and np.any(
            event_observed
        ):  # All observed events at the same time
            pass  # This case is handled by sksurv

        c_index, _, _, _, _ = concordance_index_censored(
            event_observed, all_event_times_np, all_risks_np
        )
        return c_index
    except Exception as e:  # Catch more general exceptions from sksurv if any
        print(f"Warning: Could not calculate C-index: {e}")
        return 0.0
