import os
import shutil
from typing import List


def cache_batch(wsis: List[str], dest_dir: str) -> List[str]:
    """
    Copies WSIs to a local cache directory. Handles .mrxs subdirectories if present.

    Returns:
        List[str]: Paths to copied WSIs.
    """
    os.makedirs(dest_dir, exist_ok=True)
    copied = []

    for wsi_path in wsis:
        dest_path = os.path.join(dest_dir, os.path.basename(wsi_path))
        shutil.copy(wsi_path, dest_path)
        copied.append(dest_path)

        # Handle .mrxs specific subdirectories
        if wsi_path.lower().endswith(".mrxs"):
            mrxs_dir = os.path.splitext(wsi_path)[0]
            if os.path.exists(mrxs_dir) and os.path.isdir(mrxs_dir):
                dest_mrxs_dir = os.path.join(dest_dir, os.path.basename(mrxs_dir))
                shutil.copytree(mrxs_dir, dest_mrxs_dir)

    return copied
