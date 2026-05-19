import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeAlias

import numpy as np
import torch

# Type Aliases
PathLike: TypeAlias = str | os.PathLike | Path

# Setup logger for this utility module
logger = logging.getLogger(__name__)


class JSONsaver(json.JSONEncoder):
    """
    Custom JSON Encoder to handle non-standard types like NumPy arrays,
    ranges, PyTorch dtypes, and callables for configuration saving.

    Converts unserializable types to strings or lists where appropriate.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            # Limit array size representation for sanity, but allow smaller ones to be fully dumped
            if (
                obj.size > 100 and obj.ndim > 0
            ):  # Avoid trying to dump huge arrays, but allow small ones
                return f"<NumPy Array shape={obj.shape} dtype={obj.dtype}>"
            return obj.tolist()  # Convert smaller arrays to list
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (range, Path)):  # Add Path handling
            return str(obj)
        elif obj in [
            torch.float16,
            torch.float32,
            torch.bfloat16,
            torch.int32,
            torch.int64,
        ]:  # Add more torch types
            return str(obj)
        elif callable(obj):
            try:
                # Prefer qualified name if possible
                name = getattr(obj, "__qualname__", getattr(obj, "__name__", None))
                if name:
                    module = getattr(obj, "__module__", "")
                    if module:
                        return f"<Callable: {module}.{name}>"
                    return f"<Callable: {name}>"
                # Fallback for objects without clear names (e.g., partials)
                return f"<Callable: {str(obj)}>"
            except Exception:
                return f"<Callable: {str(obj)}>"  # Safeguard
        # Let the base class default method raise the TypeError for other types
        try:
            return super().default(obj)
        except TypeError:
            return f"<Unserializable object type: {type(obj).__name__}>"


def save_json_config(
    config_path: PathLike,
    processor_instance: Optional[Any] = None,  # Can pass the Processor instance
    local_attrs: Optional[Dict[str, Any]] = None,
    ignore: Optional[List[str]] = None,
    custom_config: Optional[Dict[str, Any]] = None,  # Option to pass a pre-made dict
) -> None:
    """
    Saves configuration data to a JSON file.

    Combines attributes from a processor instance (if provided), local attributes,
    and/or a custom config dictionary. Handles non-serializable types gracefully using
    JSONsaver and filters out specified keys.

    Args:
        config_path: The full path (including filename) to save the JSON config.
        processor_instance: Optional instance (e.g., Processor) whose attributes
            should be saved.
        local_attrs: Optional dictionary of additional attributes (e.g., method parameters).
        ignore: A list of attribute names to exclude. Defaults to common Processor
                attributes like 'wsis', 'loop', 'logger', 'paths'.
        custom_config: Optional dictionary containing the configuration to save directly.
                       If provided, processor_instance and local_attrs might be ignored
                       or merged depending on implementation details (here, it merges).

    Raises:
        IOError: If the file cannot be written.
        TypeError: If JSON serialization fails unexpectedly (should be caught by JSONsaver).
    """
    config_path = Path(config_path)
    if ignore is None:
        # Default keys to ignore for the Processor class
        ignore = ["wsis", "loop", "logger", "paths"]

    config_to_save: Dict[str, Any] = {}

    # 1. Add processor instance attributes (if provided)
    if processor_instance:
        for k, v in vars(processor_instance).items():
            if k not in ignore:
                config_to_save[k] = v  # Add raw value, JSONsaver will handle type

    # 2. Add local attributes (potentially overwriting instance attributes)
    if local_attrs:
        for k, v in local_attrs.items():
            if k not in ignore:
                config_to_save[k] = v

    # 3. Add/overwrite with custom config dictionary (if provided)
    if custom_config:
        for k, v in custom_config.items():
            if k not in ignore:
                config_to_save[k] = v

    # Ensure the directory exists
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create directory for config file {config_path}: {e}")
        # Depending on severity, you might want to raise an error here
        # For now, we'll attempt to write anyway but log the error.
        # raise IOError(f"Could not create directory for config file {config_path}: {e}") from e

    # Save the combined configuration using JSONsaver
    try:
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=4, cls=JSONsaver, ensure_ascii=False)
        logger.debug(f"Configuration saved successfully to {config_path}")
    except TypeError as e:
        logger.error(
            f"JSON serialization failed for config {config_path}: {e}. Check JSONsaver handling."
        )
        # Re-raise as it indicates a fundamental issue with serialization logic
        raise
    except IOError as e:
        logger.error(f"Failed to write configuration file to {config_path}: {e}")
        # Re-raise as saving the config is often critical for reproducibility
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while saving config to {config_path}: {e}"
        )
        # Optionally re-raise depending on desired robustness
        raise
