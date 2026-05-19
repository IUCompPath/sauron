from __future__ import annotations

import json
import os
import socket
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from geopandas import gpd
from PIL import Image
from shapely import Polygon
from shapely.validation import make_valid as shapely_make_valid

try:
    from aegis.feature_extraction.wsi.factory import (
        CUCIM_EXTENSIONS,
        OPENSLIDE_EXTENSIONS,
        PIL_EXTENSIONS,
    )
except ImportError:
    # Fallback if wsi.factory is not yet available or for standalone testing
    OPENSLIDE_EXTENSIONS = {
        ".svs",
        ".tif",
        ".tiff",
        ".ndpi",
        ".vms",
        ".vmu",
        ".scn",
        ".mrxs",
        ".bif",
        ".czi",
    }
    CUCIM_EXTENSIONS = {
        ".svs",
        ".tif",
        ".tiff",
        ".czi",
        ".ndpi",
    }  # CuCIM supports more than just svs/tif
    PIL_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


ENV_aegis_HOME = "aegis_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
_cache_dir: Optional[str] = None


class CuImageWarning(UserWarning):
    """Warning for missing cucim dependency."""

    pass


# No explicit warn_cucim_missing here to avoid multiple warnings.
# The warning is handled by the `cucim` WSI object's lazy_initialize method.


def collect_valid_slides(
    wsi_dir: str,
    custom_list_path: Optional[str] = None,
    wsi_ext: Optional[List[str]] = None,
    search_nested: bool = False,
    max_workers: int = 8,
    return_relative_paths: bool = False,
    return_mpp_from_csv: bool = False,
) -> Union[
    List[str],
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str], Optional[List[float]]],
]:
    """
    Retrieve all valid WSI file paths from a directory, optionally filtered by a custom list.

    Args:
        wsi_dir (str): Path to the directory containing WSIs.
        custom_list_path (Optional[str]): Path to a CSV file with 'wsi' column of relative slide paths.
        wsi_ext (Optional[List[str]]): Allowed file extensions.
        search_nested (bool): Whether to search subdirectories.
        max_workers (int): Threads to use when checking file existence.
        return_relative_paths (bool): Whether to also return relative paths.
        return_mpp_from_csv (bool): Whether to also return MPP values from the CSV.

    Returns:
        List[str]: Full paths to valid WSIs.
        OR
        Tuple[List[str], List[str]]: (full paths, relative paths) if `return_relative_paths` is True.
        OR
        Tuple[List[str], List[str], Optional[List[float]]]: (full paths, relative paths, mpp_values) if `return_mpp_from_csv` is True.

    Raises:
        ValueError: If custom CSV is invalid or files not found.
    """
    valid_rel_paths: List[str] = []
    mpp_values: Optional[List[float]] = None

    if wsi_ext is None:
        # Default extensions if not provided
        all_extensions = (
            set(OPENSLIDE_EXTENSIONS) | set(CUCIM_EXTENSIONS) | set(PIL_EXTENSIONS)
        )
        wsi_ext = list(all_extensions)
    wsi_ext = [ext.lower() for ext in wsi_ext]

    if custom_list_path is not None:
        try:
            wsi_df = pd.read_csv(custom_list_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Custom list CSV not found at {custom_list_path}"
            ) from e

        if "wsi" not in wsi_df.columns:
            raise ValueError("Custom list CSV must contain a column named 'wsi'.")

        # Ensure 'wsi' column is string and drop NaNs
        wsi_df = wsi_df.dropna(subset=["wsi"]).astype({"wsi": str})

        # Pre-filter by extension if custom list contains many non-WSI files
        wsi_df = wsi_df[
            wsi_df["wsi"].apply(
                lambda x: any(x.lower().endswith(ext) for ext in wsi_ext)
            )
        ]

        if wsi_df.empty:
            raise ValueError(
                f"No valid slides found in the custom list at {custom_list_path} after filtering by extension."
            )

        # If custom_list_path is provided, rel_paths are exactly what's in the 'wsi' column
        rel_paths_from_csv = wsi_df["wsi"].tolist()

        # Check existence in parallel
        def exists_fn(rel_path: str) -> bool:
            return os.path.exists(os.path.join(wsi_dir, rel_path))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            existence_results = list(executor.map(exists_fn, rel_paths_from_csv))

        # Filter rel_paths_from_csv and corresponding MPP values
        final_rel_paths = []
        # Initialize final_mpp_values only if 'mpp' column exists and is requested
        final_mpp_values = (
            [] if return_mpp_from_csv and "mpp" in wsi_df.columns else None
        )

        for idx, (rel_path, exists) in enumerate(
            zip(rel_paths_from_csv, existence_results)
        ):
            if not exists:
                # Warning for missing file for nested search, means rel_path is wrong or file is truly absent.
                print(
                    f"Warning: WSI '{rel_path}' listed in CSV but not found under '{wsi_dir}'. Skipping."
                )
                continue

            final_rel_paths.append(rel_path)
            if final_mpp_values is not None:
                # It's important to use .loc and check for potential NaNs for the MPP value
                mpp_val = wsi_df.loc[wsi_df["wsi"] == rel_path, "mpp"].iloc[
                    0
                ]  # Retrieve MPP for this specific slide
                final_mpp_values.append(float(mpp_val) if pd.notna(mpp_val) else None)

        valid_rel_paths = final_rel_paths
        mpp_values = final_mpp_values

        if not valid_rel_paths:
            raise ValueError(
                f"No valid slides found in the custom list at {custom_list_path} that exist in '{wsi_dir}'."
            )

    else:  # No custom list provided, scan directory

        def matches_ext(filename: str) -> bool:
            return any(filename.lower().endswith(ext) for ext in wsi_ext)

        if search_nested:
            for root, _, files in os.walk(wsi_dir):
                for f in files:
                    if matches_ext(f):
                        rel_path = os.path.relpath(os.path.join(root, f), wsi_dir)
                        valid_rel_paths.append(rel_path)
        else:
            valid_rel_paths = [
                f
                for f in os.listdir(wsi_dir)
                if matches_ext(f)
                and os.path.isfile(os.path.join(wsi_dir, f))  # Ensure it's a file
            ]

        valid_rel_paths.sort()  # Ensure consistent order

    full_paths = [os.path.join(wsi_dir, rel) for rel in valid_rel_paths]

    if return_mpp_from_csv:
        return full_paths, valid_rel_paths, mpp_values
    elif return_relative_paths:
        return full_paths, valid_rel_paths
    else:
        return full_paths


def get_dir() -> str:
    r"""
    Get aegis cache directory used for storing downloaded models & weights.
    If :func:`~aegis.feature_extraction.utils.io.set_dir` is not called, default path is ``$aegis_HOME`` where
    environment variable ``$aegis_HOME`` defaults to ``$XDG_CACHE_HOME/aegis``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesystem layout, with a default value ``~/.cache`` if the environment
    variable is not set.
    """

    if _cache_dir is not None:
        return _cache_dir
    return _get_aegis_home()


def set_dir(d: Union[str, os.PathLike]) -> None:
    r"""
    Optionally set the aegis cache directory used to save downloaded models & weights.
    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
    global _cache_dir
    _cache_dir = os.path.expanduser(d)


def _get_aegis_home():
    aegis_home = os.path.expanduser(
        os.getenv(
            ENV_aegis_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "aegis"),
        )
    )
    return aegis_home


def has_internet_connection(timeout=3.0) -> bool:
    endpoint = os.environ.get("HF_ENDPOINT", "huggingface.co")

    if endpoint.startswith(("http://", "https://")):
        from urllib.parse import urlparse

        endpoint = urlparse(endpoint).netloc

    try:
        # Fast socket-level check
        socket.create_connection((endpoint, 443), timeout=timeout)
        return True
    except OSError:
        pass

    try:
        # Fallback HTTP-level check (if requests is available)
        import requests

        url = (
            f"https://{endpoint}"
            if not endpoint.startswith(("http://", "https://"))
            else endpoint
        )
        r = requests.head(url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def get_weights_path(model_type: str, encoder_name: str) -> str:
    """
    Retrieve the path to the weights file for a given model name.
    This function looks up the path to the weights file in a local checkpoint
    registry (checkpoints.json). If the path in the registry is absolute, it
    returns that path. If the path is relative, it joins the relative path with
    the provided weights_root directory (which is typically a sub-module path).
    Args:
        model_type (str): 'patch', 'slide', or 'seg'
        encoder_name (str): The name of the model whose weights path is to be retrieved.
    Returns:
        str: The absolute path to the weights file or directory. Returns empty string if not found locally.
    """

    assert model_type in [
        "patch",
        "slide",
        "seg",
    ], f"Model type must be 'patch', 'slide', or 'seg', not '{model_type}'"

    if model_type == "patch":
        root = os.path.join(os.path.dirname(__file__), "..", "models", "patch_encoders")
    elif model_type == "slide":
        root = os.path.join(os.path.dirname(__file__), "..", "models", "slide_encoders")
    else:  # model_type == 'seg'
        root = os.path.join(os.path.dirname(__file__), "..", "models", "segmentation")

    registry_path = os.path.join(root, "checkpoints.json")

    if not os.path.exists(registry_path):
        # This is a critical error for configured models, so raise explicitly
        raise FileNotFoundError(
            f"Model checkpoint registry not found at {registry_path}"
        )

    with open(registry_path, "r") as f:
        registry = json.load(f)

    path = registry.get(encoder_name)
    if path:
        # Priority:
        # 1. Absolute path as given in JSON
        # 2. Relative to `root` (e.g. aegis/feature_extraction/models/patch_encoders)
        # 3. Relative to `root/zoo` (common for many patch/slide models with complex structures)

        if os.path.isabs(path):
            if not os.path.exists(
                path
            ):  # If absolute path doesn't exist, return empty string for auto-download
                return ""
            return path

        # Try relative to `root`
        abs_path_candidate = os.path.abspath(os.path.join(root, path))
        if os.path.exists(abs_path_candidate):
            return abs_path_candidate

        # Try relative to `root/zoo` (common for many patch/slide models)
        abs_path_candidate_zoo = os.path.abspath(os.path.join(root, "zoo", path))
        if os.path.exists(abs_path_candidate_zoo):
            return abs_path_candidate_zoo

        # If it's a relative path in the JSON but no corresponding file is found, assume it should be auto-downloaded
        return ""

    return ""  # No path found in registry


def create_lock(path: str, suffix: Optional[str] = None):
    """
    The `create_lock` function creates a lock file to signal that a particular file or process
    is currently being worked on. This is especially useful in multiprocessing or distributed
    systems to avoid conflicts or multiple processes working on the same resource.

    Parameters:
    -----------
    path : str
        The path to the file or resource being locked.
    suffix : str, optional
        An additional suffix to append to the lock file name. This allows for creating distinct
        lock files for similar resources. Defaults to None.

    Returns:
    --------
    None
        The function creates a `.lock` file in the specified path and does not return anything.
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    lock_file = f"{path}.lock"
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(lock_file), exist_ok=True)
    with open(lock_file, "w") as f:
        f.write("")


def remove_lock(path: str, suffix: Optional[str] = None):
    """
    The `remove_lock` function removes a lock file, signaling that the file or process
    is no longer in use and is available for other operations.

    Parameters:
    -----------
    path : str
        The path to the file or resource whose lock needs to be removed.
    suffix : str, optional
        An additional suffix to identify the lock file. Defaults to None.

    Returns:
    --------
    None
        The function deletes the `.lock` file associated with the resource.
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    lock_file = f"{path}.lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)


def is_locked(path: str, suffix: Optional[str] = None):
    """
    The `is_locked` function checks if a resource is currently locked by verifying
    the existence of a `.lock` file.

    Parameters:
    -----------
    path : str
        The path to the file or resource to check for a lock.
    suffix : str, optional
        An additional suffix to identify the lock file. Defaults to None.

    Returns:
    --------
    bool
        True if the `.lock` file exists, indicating the resource is locked. False otherwise.
    """
    if suffix is not None:
        path = f"{path}_{suffix}"
    return os.path.exists(f"{path}.lock")


def update_log(path_to_log: str, key: str, message: str):
    """
    The `update_log` function appends or updates a message in a log file. It is useful for tracking
    progress or recording errors during a long-running process.

    Parameters:
    -----------
    path_to_log : str
        The path to the log file where messages will be written.
    key : str
        A unique identifier for the log entry, such as a slide name or file ID.
    message : str
        The message to log, such as a status update or error message.

    Returns:
    --------
    None
        The function writes to the log file in-place.
    """
    # Ensure log directory exists
    os.makedirs(os.path.dirname(path_to_log), exist_ok=True)

    # Read all lines to check for existing key
    lines = []
    if os.path.exists(path_to_log):
        with open(path_to_log, "r") as f:
            lines = f.readlines()

    # Rewrite log, excluding old entry for key and adding new one
    with open(path_to_log, "w") as f:
        found = False
        for line in lines:
            # Check if line starts with key + ':' to avoid partial matches
            if line.strip().startswith(f"{key}:"):
                f.write(f"{key}: {message}\n")
                found = True
            else:
                f.write(line)
        if not found:  # If key was not found, append it
            f.write(f"{key}: {message}\n")


def save_h5(
    save_path: str,
    assets: Dict[str, Any],
    attributes: Optional[Dict[str, Any]] = None,
    mode: str = "w",
):
    """
    The `save_h5` function saves a dictionary of assets to an HDF5 file. This is commonly used to store
    large datasets or hierarchical data structures in a compact and organized format.

    Parameters:
    -----------
    save_path : str
        The path where the HDF5 file will be saved.
    assets : dict
        A dictionary containing the data to save. Keys represent dataset names, and values are NumPy arrays.
    attributes : dict, optional
        A dictionary mapping dataset names to additional metadata (attributes) to save alongside the data. Defaults to None.
    mode : str, optional
        The file mode for opening the HDF5 file. Options include 'w' (write) and 'a' (append). Defaults to 'w'.

    Returns:
    --------
    None
        The function writes data and attributes to the specified HDF5 file.
    """
    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(save_path, mode) as file:
        for key, val in assets.items():
            if not isinstance(val, (np.ndarray, torch.Tensor)):
                raise TypeError(
                    f"Asset '{key}' must be a NumPy array or PyTorch tensor, but got {type(val)}"
                )

            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()  # Ensure it's a NumPy array

            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                # Handle empty array/scalar chunking
                chunk_shape = (1,) + data_shape[1:] if data_shape else (1,)
                maxshape = (None,) + data_shape[1:] if data_shape else (None,)

                # Special handling for string data type if needed, but for numerical/bool h5py handles automatically
                # if data_type == np.object_: # E.g., for strings, use h5py.string_dtype
                #    data_type = h5py.string_dtype(encoding='utf-8')

                dset = file.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=maxshape,
                    chunks=chunk_shape,
                    dtype=data_type,
                )
                dset[:] = val
                if attributes is not None:
                    if key in attributes.keys():
                        for attr_key, attr_val in attributes[key].items():
                            try:
                                # Serialize if the attribute value is a dictionary
                                if isinstance(attr_val, dict):
                                    attr_val = json.dumps(attr_val)
                                # Serialize Nones to string 'None'
                                elif attr_val is None:
                                    attr_val = "None"
                                dset.attrs[attr_key] = attr_val
                            except Exception as e:
                                print(
                                    f"WARNING: Could not save attribute {attr_key} with value {attr_val} for asset {key}: {e}"
                                )

            else:  # Dataset already exists, append data
                dset = file[key]
                # Ensure compatibility before resizing and appending
                if dset.ndim != val.ndim:
                    raise ValueError(
                        f"Cannot append to dataset '{key}'. Dimensionality mismatch: existing {dset.ndim} vs new {val.ndim}."
                    )
                if dset.ndim > 1 and dset.shape[1:] != val.shape[1:]:
                    raise ValueError(
                        f"Cannot append to dataset '{key}'. Shape mismatch: existing {dset.shape[1:]} vs new {val.shape[1:]}."
                    )

                dset.resize(len(dset) + data_shape[0], axis=0)
                if dset.dtype != val.dtype:
                    # Attempt type conversion if possible, otherwise raise error
                    try:
                        val_converted = val.astype(dset.dtype)
                        dset[-data_shape[0] :] = val_converted
                    except TypeError:
                        raise TypeError(
                            f"Cannot append to dataset '{key}'. Data type mismatch and automatic conversion failed: existing {dset.dtype} vs new {val.dtype}."
                        )
                else:
                    dset[-data_shape[0] :] = val


def read_coords(coords_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    The `read_coords` function reads patch coordinates from an HDF5 file, along with any user-defined
    attributes stored during the patching process. This function is essential for workflows that rely
    on spatial metadata, such as patch-based analysis in computational pathology.

    Parameters:
    -----------
    coords_path : str
        The path to the HDF5 file containing patch coordinates and attributes.

    Returns:
    --------
    attrs : dict
        A dictionary of user-defined attributes stored during patching.
    coords : np.array
        An array of patch coordinates at level 0.
    """
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Coordinate file not found: {coords_path}")

    with h5py.File(coords_path, "r") as f:
        if "coords" not in f:
            raise ValueError(f"Dataset 'coords' not found in HDF5 file: {coords_path}")

        attrs = dict(f["coords"].attrs)
        coords = f["coords"][:]
    return attrs, coords


def read_coords_legacy(coords_path: str) -> Tuple[int, int, int, np.ndarray]:
    """
    The `read_coords_legacy` function reads legacy patch coordinates from an HDF5 file. This function
    is designed for compatibility with older patching tools such as CLAM or Fishing-Rod, which used
    a different structure for storing patching metadata.

    Parameters:
    -----------
    coords_path : str
        The path to the HDF5 file containing legacy patch coordinates and metadata.

    Returns:
    --------
    patch_size : int
        The target patch size at the desired magnification.
    patch_level : int
        The patch level used when reading the slide.
    custom_downsample : int
        Any additional downsampling applied to the patches.
    coords : np.array
        An array of patch coordinates.
    """
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Legacy coordinate file not found: {coords_path}")

    with h5py.File(coords_path, "r") as f:
        if "coords" not in f:
            raise ValueError(
                f"Dataset 'coords' not found in legacy HDF5 file: {coords_path}"
            )

        patch_size = f["coords"].attrs["patch_size"]
        patch_level = f["coords"].attrs["patch_level"]
        # Default to 1 if 'custom_downsample' attribute is missing
        custom_downsample = f["coords"].attrs.get("custom_downsample", 1)
        coords = f["coords"][:]
    return patch_size, patch_level, custom_downsample, coords


def mask_to_gdf(
    mask: np.ndarray,
    keep_ids: List[int] = [],
    exclude_ids: List[int] = [],
    max_nb_holes: int = 0,
    min_contour_area: float = 1000,
    pixel_size: float = 1,
    contour_scale: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Convert a binary mask into a GeoDataFrame of polygons representing detected regions.

    This function processes a binary mask to identify contours, filter them based on specified parameters,
    and scale them to the desired dimensions. The output is a GeoDataFrame where each row corresponds
    to a detected region, with polygons representing the tissue contours and their associated holes.

    Args:
        mask (np.ndarray): The binary mask to process, where non-zero regions represent areas of interest.
        keep_ids (List[int], optional): A list of contour indices to keep. Defaults to an empty list (keep all).
        exclude_ids (List[int], optional): A list of contour indices to exclude. Defaults to an empty list.
        max_nb_holes (int, optional): The maximum number of holes to retain for each contour.
            Use 0 to retain no holes. Defaults to 0.
        min_contour_area (float, optional): Minimum area (in pixels) for a contour to be retained. Defaults to 1000.
        pixel_size (float, optional): Pixel size of level 0. Defaults to 1.
        contour_scale (float, optional): Scaling factor for the output polygons. Defaults to 1.0.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons for the detected regions. The GeoDataFrame
        includes a `tissue_id` column (integer ID for each region) and a `geometry` column (polygons).

    Raises:
        Exception: If no valid contours are detected in the mask.

    Notes:
        - The function internally downsamples the input mask for efficiency before finding contours.
        - The resulting polygons are scaled back to the original resolution using the `contour_scale` parameter.
        - Holes in contours are also detected and included in the resulting polygons.
    """

    TARGET_EDGE_SIZE = 2000
    scale = (
        TARGET_EDGE_SIZE / mask.shape[0] if mask.shape[0] > 0 else 1.0
    )  # Avoid division by zero for empty masks

    downscaled_mask = cv2.resize(
        mask, (round(mask.shape[1] * scale), round(mask.shape[0] * scale))
    )

    # Find and filter contours
    # RETR_TREE: retrieves all contours and reconstructs a full hierarchy of nested contours.
    # RETR_CCOMP: retrieves all of the contours and organizes them into a two-level hierarchy.
    #             At the first level are the external boundaries of the components.
    #             At the second level are the boundaries of the holes inside those components.
    mode = cv2.RETR_TREE if max_nb_holes == 0 else cv2.RETR_CCOMP
    contours, hierarchy = cv2.findContours(downscaled_mask, mode, cv2.CHAIN_APPROX_NONE)

    if hierarchy is None:  # No contours found
        hierarchy = np.array([])
    else:
        hierarchy = np.squeeze(hierarchy, axis=(0,))[
            :, 2:
        ]  # Remove batch dim and keep parent/child indices

    filter_params = {
        "filter_color_mode": "none",  # Not used in this function's logic
        "max_n_holes": max_nb_holes,
        "a_t": min_contour_area * pixel_size**2,  # Minimum area in original pixels^2
        "min_hole_area": 4000 * pixel_size**2,  # Minimum hole area in original pixels^2
    }

    foreground_contours, hole_contours = filter_contours(
        contours, hierarchy, filter_params
    )

    if len(foreground_contours) == 0:
        print("[Warning] No contour were detected. Contour GeoJSON will be empty.")
        return gpd.GeoDataFrame(columns=["tissue_id", "geometry"])
    else:
        # Scale contours back to level 0 (original resolution)
        # The scale applied here is `(original_pixels / target_pixels_in_mask) * (output_geojson_scale / original_mpp_scale)`
        # `contour_scale` is typically 1.0. `scale` is target_edge_size / mask_height.
        # So overall factor is `1 / scale = original_height / target_edge_size`
        final_scale_factor = (
            contour_scale / scale
        )  # Example: 1.0 / (2000 / original_mask_height)

        contours_tissue = scale_contours(
            foreground_contours, final_scale_factor, is_nested=False
        )
        contours_holes = scale_contours(
            hole_contours, final_scale_factor, is_nested=True
        )

    if len(keep_ids) > 0:
        contour_ids = set(keep_ids) - set(exclude_ids)
    else:
        # Default to all found contours if no specific IDs are kept/excluded
        contour_ids = set(np.arange(len(contours_tissue))) - set(exclude_ids)

    tissue_ids = [i for i in sorted(list(contour_ids))]  # Ensure consistent order
    polygons = []
    for i in tissue_ids:
        # Ensure contour is valid before creating polygon
        if i >= len(contours_tissue):  # Skip if contour_id is out of bounds
            continue

        holes = (
            [contours_holes[i][j].squeeze(1) for j in range(len(contours_holes[i]))]
            if len(contours_holes[i]) > 0
            else None
        )

        # Ensure exterior coords are at least 3 points for a valid polygon
        exterior_coords = contours_tissue[i].squeeze(1)
        if len(exterior_coords) < 3:
            continue  # Skip invalid contour

        polygon = Polygon(exterior_coords, holes=holes)
        if not polygon.is_valid:
            polygon = fix_invalid_polygon(
                polygon
            )  # Use the shapely fix_invalid_polygon

        if polygon.is_empty:  # After fixing, it might become empty
            continue

        polygons.append(polygon)

    # Create GeoDataFrame with tissue_id and geometry
    gdf_contours = gpd.GeoDataFrame(
        pd.DataFrame(tissue_ids[: len(polygons)], columns=["tissue_id"]),
        geometry=polygons,
    )

    # Set a Coordinate Reference System (CRS) - Web Mercator is common for geospatial data visualization
    gdf_contours.set_crs("EPSG:3857", inplace=True, allow_override=True)

    return gdf_contours


def filter_contours(
    contours: List[np.ndarray], hierarchy: np.ndarray, filter_params: Dict[str, float]
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """
    The `filter_contours` function processes a list of contours and their hierarchy, filtering
    them based on specified criteria such as minimum area and hole limits. This function is
    typically used in digital pathology workflows to isolate meaningful tissue regions.

    Original implementation from: https://github.com/mahmoodlab/CLAM/blob/f1e93945d5f5ac6ed077cb020ed01cf984780a77/wsi_core/WholeSlideImage.py#L97

    Parameters:
    -----------
    contours : list
        A list of contours representing detected regions.
    hierarchy : np.ndarray
        The hierarchy of the contours, used to identify relationships (e.g., parent-child).
    filter_params : dict
        A dictionary containing filtering criteria. Expected keys include:
        - `max_n_holes`: Maximum number of holes to retain.
        - `a_t`: Minimum area threshold for contours (in pixel_size units squared).
        - `min_hole_area`: Minimum area threshold for holes (in pixel_size units squared).

    Returns:
    --------
    tuple:
        A tuple containing:
        - Filtered foreground contours (list)
        - Corresponding hole contours (list)
    """
    if not hierarchy.size:
        return [], []

    # Find indices of foreground contours (parent == -1)
    foreground_indices = np.flatnonzero(hierarchy[:, 1] == -1)
    filtered_foregrounds = []
    filtered_holes = []

    # Loop through each foreground contour
    for cont_idx in foreground_indices:
        contour = contours[cont_idx]
        hole_indices = np.flatnonzero(hierarchy[:, 1] == cont_idx)

        # Calculate area of the contour (foreground area minus holes)
        contour_area = cv2.contourArea(contour)
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in hole_indices]

        # Area is already scaled by pixel_size^2 in mask_to_gdf, so just use raw contour_area
        net_area = contour_area - sum(hole_areas)  # Already in target_edge_size pixels

        # Skip contours with negligible area (already converted by pixel_size in mask_to_gdf)
        if net_area <= 0 or net_area <= filter_params["a_t"]:
            continue

        # Append valid contours
        filtered_foregrounds.append(contour)

        # Filter and limit the number of holes
        valid_holes = [
            contours[hole_idx]
            for hole_idx in hole_indices
            if cv2.contourArea(contours[hole_idx])
            > filter_params["min_hole_area"]  # Area check against mask_pixels
        ]
        valid_holes = sorted(valid_holes, key=cv2.contourArea, reverse=True)[
            : filter_params["max_n_holes"]
        ]
        filtered_holes.append(valid_holes)

    return filtered_foregrounds, filtered_holes


def fix_invalid_polygon(polygon: Polygon) -> Polygon:
    """
    The `fix_invalid_polygon` function attempts to fix invalid polygons by applying small buffer operations.
    This is particularly useful in cases where geometric operations result in self-intersecting
    or malformed polygons.

    Parameters:
    -----------
    polygon : shapely.geometry.Polygon
        The input polygon that may be invalid.

    Returns:
    --------
    shapely.geometry.Polygon
        A valid polygon object.

    Raises:
    -------
    ValueError:
        If the function fails to create a valid polygon after several attempts.
    """

    # Try using shapely's built-in make_valid first, which is often sufficient
    if not polygon.is_valid:
        new_polygon = shapely_make_valid(polygon)
        if isinstance(new_polygon, Polygon) and new_polygon.is_valid:
            return new_polygon

    for i in [0, 0.1, -0.1, 0.2]:
        new_polygon = polygon.buffer(i)
        if isinstance(new_polygon, Polygon) and new_polygon.is_valid:
            return new_polygon

    # Fallback to a warning and returning potentially invalid or empty polygon if all attempts fail
    # Or, raise an error if strict validity is required.
    warnings.warn(
        "Failed to make a valid polygon after multiple attempts. Original polygon might be too complex or malformed."
    )
    return polygon  # Return the last attempt or original polygon


def scale_contours(
    contours: Union[List[np.ndarray], List[List[np.ndarray]]],
    scale: float,
    is_nested: bool = False,
) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
    """
    The `scale_contours` function scales the dimensions of contours or nested contours (e.g., holes)
    by a specified factor. This is useful for resizing detected regions in masks or GeoDataFrames.

    Parameters:
    -----------
    contours : list
        A list of contours (or nested lists for holes) to be scaled.
    scale : float
        The scaling factor to apply.
    is_nested : bool, optional
        Indicates whether the input is a nested list of contours (e.g., for holes). Defaults to False.

    Returns:
    --------
    list:
        A list of scaled contours or nested contours.
    """
    if is_nested:
        return [
            [np.array(hole * scale, dtype=np.int32) for hole in holes]
            for holes in contours
        ]
    return [np.array(cont * scale, dtype=np.int32) for cont in contours]


def overlay_gdf_on_thumbnail(
    gdf_contours: gpd.GeoDataFrame,
    thumbnail: np.ndarray,
    contours_saveto: str,
    scale: float,
    tissue_color: Tuple[int, int, int] = (0, 255, 0),
    hole_color: Tuple[int, int, int] = (255, 0, 0),
):
    """
    The `overlay_gdf_on_thumbnail` function overlays polygons from a GeoDataFrame onto a scaled
    thumbnail image using OpenCV. This is particularly useful for visualizing tissue regions and
    their boundaries on smaller representations of whole-slide images.

    Parameters:
    -----------
    gdf_contours : gpd.GeoDataFrame
        A GeoDataFrame containing the polygons to overlay, with a `geometry` column.
    thumbnail : np.ndarray
        The thumbnail image as a NumPy array (RGB or BGR).
    contours_saveto : str
        The file path to save the annotated thumbnail.
    scale : float
        The scaling factor between the GeoDataFrame coordinates and the thumbnail resolution.
    tissue_color : tuple, optional
        The color (BGR format) for tissue polygons. Defaults to green `(0, 255, 0)`.
    hole_color : tuple, optional
        The color (BGR format) for hole polygons. Defaults to red `(255, 0, 0)`.

    Returns:
    --------
    None
        The function saves the annotated image to the specified file path.
    """
    # Ensure thumbnail is mutable
    annotated_thumbnail = np.copy(thumbnail)

    # Convert to BGR if it's RGB (OpenCV uses BGR by default for drawing)
    if (
        annotated_thumbnail.shape[2] == 3
        and annotated_thumbnail[0, 0, 0] > annotated_thumbnail[0, 0, 2]
    ):  # Simple check for RGB vs BGR
        annotated_thumbnail = cv2.cvtColor(annotated_thumbnail, cv2.COLOR_RGB2BGR)

    for poly in gdf_contours.geometry:
        if poly.is_empty:
            continue

        # Draw tissue boundary
        if poly.exterior:
            exterior_coords = (np.array(poly.exterior.coords) * scale).astype(np.int32)
            # Reshape to (N, 1, 2) for cv2.polylines
            cv2.polylines(
                annotated_thumbnail,
                [exterior_coords.reshape(-1, 1, 2)],
                isClosed=True,
                color=tissue_color,
                thickness=2,
            )

        # Draw holes (if any) in a different color
        if poly.interiors:
            for interior in poly.interiors:
                interior_coords = (np.array(interior.coords) * scale).astype(np.int32)
                cv2.polylines(
                    annotated_thumbnail,
                    [interior_coords.reshape(-1, 1, 2)],
                    isClosed=True,
                    color=hole_color,
                    thickness=2,
                )

    # Crop black borders of the annotated image (assuming black background if it was filled)
    # This might be too aggressive if the thumbnail itself has black regions
    gray_thumbnail = cv2.cvtColor(annotated_thumbnail, cv2.COLOR_BGR2GRAY)
    nz = np.nonzero(gray_thumbnail)  # Non-zero pixel locations
    if nz[0].size > 0:  # Check if there are any non-zero pixels
        ymin, ymax = np.min(nz[0]), np.max(nz[0])
        xmin, xmax = np.min(nz[1]), np.max(nz[1])
        cropped_annotated = annotated_thumbnail[ymin:ymax, xmin:xmax]
    else:
        cropped_annotated = annotated_thumbnail  # No non-zero pixels, keep as is

    # Save the annotated image
    os.makedirs(os.path.dirname(contours_saveto), exist_ok=True)
    # Convert back to RGB for PIL-compatible saving if it was BGR for drawing
    cropped_annotated = cv2.cvtColor(cropped_annotated, cv2.COLOR_BGR2RGB)
    Image.fromarray(cropped_annotated).save(contours_saveto)


def get_num_workers(
    batch_size: int,
    factor: float = 0.75,
    fallback: int = 16,
    max_workers: int | None = None,
) -> int:
    """
    The `get_num_workers` function calculates the optimal number of workers for a PyTorch DataLoader,
    balancing system resources and workload. This ensures efficient data loading while avoiding
    resource overutilization.

    Parameters:
    -----------
    batch_size : int
        The batch size for the DataLoader.
    factor : float, optional
        The fraction of available CPU cores to use. Defaults to 0.75 (75% of available cores).
    fallback : int, optional
        The default number of workers to use if the system's CPU core count cannot be determined. Defaults to 16.
    max_workers : int or None, optional
        The maximum number of workers allowed. Defaults to `2 * batch_size` if not provided.

    Returns:
    --------
    int
        The calculated number of workers for the DataLoader.

    Example:
    --------
    >>> num_workers = get_num_workers(batch_size=64, factor=0.5)
    >>> print(num_workers)
    8

    Notes:
    ------
    - The number of workers is clipped to a minimum of 1 to ensure multiprocessing is not disabled.
    - The maximum number of workers defaults to `2 * batch_size` unless explicitly specified.
    - The function ensures compatibility with systems where `os.cpu_count()` may return `None`.
    - On Windows systems, the number of workers is always set to 0 to ensure compatibility with PyTorch datasets whose attributes may not be serializable.
    """

    # Disable pytorch multiprocessing on Windows
    if os.name == "nt":
        return 0

    num_cores = os.cpu_count() or fallback
    num_workers_calculated = int(
        factor * num_cores
    )  # Use a fraction of available cores

    # Default max_workers to 2 * batch_size if not specified
    max_workers_effective = max_workers if max_workers is not None else (2 * batch_size)

    num_workers = np.clip(num_workers_calculated, 1, max_workers_effective)
    return int(num_workers)
