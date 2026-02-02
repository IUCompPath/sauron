"""
Manages data splits for different task types.
"""

import argparse
from typing import TYPE_CHECKING

from aegis.training.param_builders import safe_getattr
from aegis.training.task_utils import is_classification, is_survival

if TYPE_CHECKING:
    from aegis.data.classMILDataset import ClassificationDataManager
    from aegis.data.survMILDataset import SurvivalDataManager


class SplitManager:
    """Manages data splits for different task types."""

    def __init__(self, args: argparse.Namespace):
        """
        Initialize SplitManager with arguments.

        Args:
            args: Argument namespace
        """
        self.args = args

    def create_splits(
        self, data_manager: "ClassificationDataManager | SurvivalDataManager"
    ) -> range:
        """
        Create splits in the data manager and return the fold range.

        Args:
            data_manager: DataManager instance (Classification or Survival)

        Returns:
            Range object for fold iteration

        Raises:
            ValueError: If task type is not supported or split creation fails
        """
        if is_classification(self.args):
            return self._create_classification_splits(data_manager)
        elif is_survival(self.args):
            return self._create_survival_splits(data_manager)
        else:
            raise ValueError(
                f"Task type {self.args.task_type} split creation not defined."
            )

    def _create_classification_splits(
        self, data_manager: "ClassificationDataManager"
    ) -> range:
        """Create splits for classification task."""
        data_manager.create_k_fold_splits(
            num_folds=self.args.k,
            test_set_size=safe_getattr(self.args, "test_frac", 0.1),
        )
        num_actual_folds = data_manager.get_number_of_folds()

        if num_actual_folds == 0 and self.args.k > 0:
            print(
                "No K-folds generated (num_folds=0 in DataManager), "
                "running as single train/test split if test_frac > 0."
            )
            if self.args.k <= 1:
                print(f"Running a single train/val/test split (args.k={self.args.k}).")
                return range(1)
            else:
                raise ValueError(
                    f"args.k={self.args.k} but DataManager created 0 folds. "
                    "Check data or split logic."
                )
        else:
            return range(self.args.k_start, min(self.args.k_end, num_actual_folds))

    def _create_survival_splits(self, data_manager: "SurvivalDataManager") -> range:
        """Create splits for survival task."""
        data_manager.create_splits_from_generating_function(
            k=self.args.k,
            val_num=safe_getattr(self.args, "val_num_survival", (0.15, 0.15)),
            test_num=safe_getattr(self.args, "test_num_survival", (0.15, 0.15)),
            label_frac=safe_getattr(self.args, "label_frac", 1.0),
            custom_test_ids=safe_getattr(self.args, "custom_test_ids", None),
        )

        # If the manager knows how many folds were actually generated (e.g. 1 for manual splits), respect that.
        num_generated = getattr(data_manager, "num_folds_generated", self.args.k)
        return range(self.args.k_start, min(self.args.k_end, num_generated))

    def set_current_fold(
        self,
        data_manager: "ClassificationDataManager | SurvivalDataManager",
        fold_idx: int,
    ) -> bool:
        """
        Set the current fold in the data manager.

        Args:
            data_manager: DataManager instance
            fold_idx: Index of the fold to set

        Returns:
            True if fold was set successfully, False otherwise
        """
        if is_classification(self.args):
            data_manager.set_current_fold(fold_index=fold_idx)
            return True
        elif is_survival(self.args):
            start_from_fold = fold_idx if fold_idx == self.args.k_start else None
            success = data_manager.set_next_fold_from_generator(
                start_from_fold=start_from_fold
            )
            if not success:
                print(
                    f"SurvivalDataManager's split generator exhausted before reaching fold {fold_idx}."
                )
            return success
        else:
            raise ValueError(
                f"Task type {self.args.task_type} fold setting not defined."
            )
