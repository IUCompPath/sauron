"""
Constants and enums for training tasks.
"""

from enum import Enum
from typing import List


class TaskType(str, Enum):
    """Supported task types for MIL training."""

    CLASSIFICATION = "classification"
    SURVIVAL = "survival"

    @classmethod
    def from_string(cls, value: str) -> "TaskType":
        """Convert string to TaskType enum."""
        value_lower = value.lower()
        for task_type in cls:
            if task_type.value == value_lower:
                return task_type
        raise ValueError(
            f"Unknown task type: {value}. Must be one of {[t.value for t in cls]}"
        )


# Metric keys for each task type
CLASSIFICATION_METRIC_KEYS: List[str] = [
    "test_auc",
    "val_auc",
    "test_acc",
    "val_acc",
]

SURVIVAL_METRIC_KEYS: List[str] = [
    "test_c_index",
    "val_c_index",
]


def get_metric_keys_for_task(task_type: str) -> List[str]:
    """
    Returns the expected metric keys for a given task type.

    Args:
        task_type: Task type string (case-insensitive)

    Returns:
        List of metric key strings expected from train_fold

    Raises:
        ValueError: If task_type is not supported
    """
    task_type_enum = TaskType.from_string(task_type)
    if task_type_enum == TaskType.CLASSIFICATION:
        return CLASSIFICATION_METRIC_KEYS.copy()
    elif task_type_enum == TaskType.SURVIVAL:
        return SURVIVAL_METRIC_KEYS.copy()
    else:
        raise ValueError(f"Unknown task_type: {task_type} for defining metrics.")
