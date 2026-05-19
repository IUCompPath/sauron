"""
Utility functions for task type handling.
"""

import argparse

from aegis.training.constants import TaskType


def get_task_type(args: argparse.Namespace) -> TaskType:
    """
    Get the task type from args as a TaskType enum.

    Args:
        args: Argument namespace containing task_type

    Returns:
        TaskType enum value
    """
    return TaskType.from_string(args.task_type)


def is_classification(args: argparse.Namespace) -> bool:
    """
    Check if the task type is classification.

    Args:
        args: Argument namespace containing task_type

    Returns:
        True if task type is classification, False otherwise
    """
    return get_task_type(args) == TaskType.CLASSIFICATION


def is_survival(args: argparse.Namespace) -> bool:
    """
    Check if the task type is survival.

    Args:
        args: Argument namespace containing task_type

    Returns:
        True if task type is survival, False otherwise
    """
    return get_task_type(args) == TaskType.SURVIVAL


def get_data_manager_type(args: argparse.Namespace) -> str:
    """
    Get the task type as a lowercase string (for backward compatibility).

    Args:
        args: Argument namespace containing task_type

    Returns:
        Lowercase task type string
    """
    return get_task_type(args).value
