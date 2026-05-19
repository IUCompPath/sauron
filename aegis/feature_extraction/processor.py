from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeAlias

import geopandas as gpd
import torch
from tqdm import tqdm

from aegis.feature_extraction.models.patch_encoders.factory import (
    BasePatchEncoder,
)
from aegis.feature_extraction.models.patch_encoders.factory import (
    encoder_factory as patch_encoder_factory,
)  # For type hinting
from aegis.feature_extraction.models.segmentation.factory import (
    SegmentationModel,
)  # For type hinting
from aegis.feature_extraction.models.slide_encoders.factory import (
    BaseSlideEncoder,
    slide_to_patch_encoder_name,
)
from aegis.feature_extraction.utils.config import JSONsaver
from aegis.feature_extraction.utils.io import (
    collect_valid_slides,
    create_lock,
    is_locked,
    remove_lock,
    update_log,
)
from aegis.feature_extraction.wsi.factory import (
    OPENSLIDE_EXTENSIONS,
    PIL_EXTENSIONS,
    WSIReaderType,
    load_wsi,
)

# --- Setup Basic Logging ---
logger = logging.getLogger(__name__)

# --- Type Aliases for Clarity ---
PathLike: TypeAlias = str | os.PathLike


class Processor:
    def __init__(
        self,
        job_dir: PathLike,
        wsi_source: PathLike,
        wsi_ext: Optional[List[str]] = None,
        wsi_cache: Optional[PathLike] = None,
        clear_cache: bool = False,
        skip_errors: bool = False,
        custom_mpp_keys: Optional[List[str]] = None,
        custom_list_of_wsis: Optional[PathLike] = None,
        max_workers: Optional[int] = None,  # Used by dataloaders for num_workers
        reader_type: Optional[WSIReaderType] = None,
        search_nested: bool = False,
    ) -> None:
        """
        The `Processor` class handles all preprocessing steps starting from whole-slide images (WSIs).

        Available methods:
            - `run_segmentation_job`: Performs tissue segmentation on all slides managed by the processor.
            - `run_patching_job`: Extracts patch coordinates from the segmented tissue regions of slides.
            - `run_patch_feature_extraction_job`: Extracts patch-level features using a specified patch encoder.
            - `run_slide_feature_extraction_job`: Extracts slide-level features using a specified slide encoder.

        Parameters:
            job_dir (PathLike):
                The directory where the results of processing, including segmentations, patches, and extracted features,
                will be saved. This should be an existing directory with sufficient storage.
            wsi_source (PathLike):
                The directory containing the WSIs to be processed. This can either be a local directory
                or a network-mounted drive. All slides in this directory matching the specified file
                extensions will be considered for processing.
            wsi_ext (List[str]):
                A list of accepted WSI file extensions, such as ['.ndpi', '.svs']. This allows for
                filtering slides based on their format. If set to None, a default list of common extensions
                will be used. Defaults to None.
            wsi_cache (PathLike, optional):
                An optional directory for caching WSIs locally. If specified, slides will be copied
                from the source directory to this local directory before processing, improving performance
                when the source is a network drive. Defaults to None.
            clear_cache (bool, optional):
                A flag indicating whether slides in the cache should be deleted after processing.
                This helps manage storage space. Defaults to False.
            skip_errors (bool, optional):
                A flag specifying whether to continue processing if an error occurs on a slide.
                If set to False, the process will stop on the first error. Defaults to False.
            custom_mpp_keys (List[str], optional):
                A list of custom keys in the slide metadata for retrieving the microns per pixel (MPP) value.
                If not provided, standard keys will be used. Defaults to None.
            custom_list_of_wsis (PathLike, optional):
                Path to a csv file with a custom list of WSIs to process in a field called 'wsi' (including extensions). If provided, only
                these slides will be considered for processing. Defaults to None, which means all
                slides matching the wsi_ext extensions will be processed.
                Note: If `custom_list_of_wsis` is provided, any names that do not match the available slides will be ignored, and a warning will be printed.
            max_workers (int, optional):
                Maximum number of workers for data loading. If None, the default behavior will be used.
                Defaults to None.
            reader_type (WSIReaderType, optional):
                Force the image reader engine to use. Options are are ["openslide", "image", "cucim"]. Defaults to None
                (auto-determine the right engine based on image extension).
            search_nested (bool, optional):
                If True, the processor will recursively search for WSIs within all subdirectories of `wsi_source`.
                All matching files (based on `wsi_ext`) found at any depth within the directory
                tree will be included. Each slide will be identified by its relative path to `wsi_source`, but only
                the filename (excluding directory structure) will be used for downstream outputs (e.g., segmentation filenames).
                If False, only files directly inside `wsi_source` will be considered.
                Defaults to False.


        Returns:
            None: This method initializes the class instance and sets up the environment for processing.

        Example
        -------
        Initialize the `Processor` for a directory of WSIs:

        >>> processor = Processor(
        ...     job_dir="results/",
        ...     wsi_source="data/slides/",
        ...     wsi_ext=[".svs", ".ndpi"],
        ... )
        >>> print(f"Processor initialized for {len(processor.wsis)} slides.")

        Raises:
            AssertionError: If `wsi_ext` is not a list or if any extension does not start with a period.
        """

        if not (sys.version_info.major >= 3 and sys.version_info.minor >= 9):
            raise EnvironmentError(
                "aegis requires Python 3.9 or above. Python 3.10 is recommended."
            )

        self.job_dir = os.path.abspath(job_dir)
        self.wsi_source = os.path.abspath(wsi_source)
        self.wsi_ext = wsi_ext or (list(PIL_EXTENSIONS) + list(OPENSLIDE_EXTENSIONS))
        self.wsi_cache = os.path.abspath(wsi_cache) if wsi_cache else None
        self.clear_cache = clear_cache
        self.skip_errors = skip_errors
        self.custom_mpp_keys = custom_mpp_keys
        self.max_workers = max_workers
        self.reader_type = reader_type
        self.search_nested = search_nested

        # Validate extensions
        assert isinstance(
            self.wsi_ext, list
        ), f"wsi_ext must be a list, got {type(self.wsi_ext)}"
        for ext in self.wsi_ext:
            assert ext.startswith(
                "."
            ), f"Invalid extension: {ext} (must start with a period)"

        # === Collect slide paths and relative paths ===
        full_paths, rel_paths, mpp_values_from_csv = collect_valid_slides(
            wsi_dir=wsi_source,
            custom_list_path=custom_list_of_wsis,
            wsi_ext=self.wsi_ext,
            search_nested=search_nested,
            max_workers=max_workers,
            return_mpp_from_csv=True,  # New return
            return_relative_paths=True,
        )

        self.wsi_rel_paths = rel_paths if custom_list_of_wsis else None

        logger.info(
            f"[PROCESSOR] Found {len(full_paths)} valid slides in {wsi_source}."
        )

        # === Initialize WSIs ===
        self.wsis = []
        init_log_path = os.path.join(self.job_dir, "_processor_init_log.txt")
        for wsi_idx, abs_path in enumerate(full_paths):
            name = os.path.basename(
                abs_path
            )  # Name for output files is just the filename, not full relative path
            original_full_path = full_paths[
                wsi_idx
            ]  # The true path of the WSI file on disk

            # Use original_full_path to determine if caching is needed
            load_path = (
                os.path.join(self.wsi_cache, name)
                if self.wsi_cache
                else original_full_path
            )

            tissue_seg_path = os.path.join(
                self.job_dir,
                "segmentation_results",
                "contours_geojson",
                f"{os.path.splitext(name)[0]}.geojson",
            )
            if not os.path.exists(tissue_seg_path):
                tissue_seg_path = None

            try:
                slide = load_wsi(
                    slide_path=load_path,  # Path where WSI is expected to be *loaded from*
                    original_path=original_full_path,  # Original source path for caching logic
                    name=name,  # Base filename for output file naming
                    tissue_seg_path=tissue_seg_path,
                    custom_mpp_keys=self.custom_mpp_keys,
                    mpp=(
                        mpp_values_from_csv[wsi_idx]
                        if mpp_values_from_csv is not None
                        else None
                    ),
                    max_workers=self.max_workers,
                    reader_type=self.reader_type,
                    lazy_init=True,
                )
                self.wsis.append(slide)
                update_log(init_log_path, name, "INFO - WSI object created")

            except Exception as e:
                message = f"ERROR creating WSI object for {name} at {load_path} (original: {original_full_path}): {e}"
                update_log(init_log_path, name, message)
                if self.skip_errors:
                    logger.error(message)
                else:
                    raise RuntimeError(message) from e

    def _get_job_paths(self, job_name: str, sub_dirs: List[str]) -> Dict[str, str]:
        """Helper to create and return paths for a specific processing job."""
        base_dir = os.path.join(self.job_dir, job_name)
        paths = {"base": base_dir}
        for sub in sub_dirs:
            path = os.path.join(base_dir, sub)
            os.makedirs(path, exist_ok=True)
            paths[sub] = path
        paths["config"] = os.path.join(base_dir, f"_config_{job_name}.json")
        paths["log"] = os.path.join(base_dir, f"_log_{job_name}.txt")
        return paths

    def populate_cache(self, start_idx: int = 0) -> None:
        """
        Copies WSI files from the source directory to the local cache directory.
        Only copies slides from `start_idx` onwards.
        """
        if not self.wsi_cache:
            logger.info("No cache directory specified. Skipping cache population.")
            return

        cache_log_path = os.path.join(self.wsi_cache, "_cache_log.txt")
        logger.info(f"Populating cache directory: {self.wsi_cache}")

        # Filter slides to process based on start_idx
        wsis_to_process = self.wsis[start_idx:]

        progress_bar = tqdm(
            wsis_to_process,
            desc="Populating cache",
            total=len(wsis_to_process),
            unit="slide",
        )

        for wsi in progress_bar:
            slide_fullname = wsi.name + wsi.ext
            cache_file_path = os.path.join(self.wsi_cache, slide_fullname)
            source_file_path = wsi.original_path  # Use the true source path

            progress_bar.set_postfix_str(f"{slide_fullname}")

            if os.path.exists(cache_file_path) and not is_locked(cache_file_path):
                update_log(cache_log_path, slide_fullname, "INFO - Already in cache")
                continue

            if is_locked(cache_file_path):
                update_log(
                    cache_log_path, slide_fullname, "SKIP - Locked by another process"
                )
                continue

            try:
                create_lock(cache_file_path)
                update_log(cache_log_path, slide_fullname, "LOCK - Copying")
                shutil.copy2(source_file_path, cache_file_path)
                if source_file_path.lower().endswith(".mrxs"):
                    mrxs_dir = os.path.splitext(source_file_path)[0]
                    if os.path.exists(mrxs_dir) and os.path.isdir(mrxs_dir):
                        dest_mrxs_dir = os.path.join(
                            self.wsi_cache, os.path.basename(mrxs_dir)
                        )
                        shutil.copytree(mrxs_dir, dest_mrxs_dir)
                update_log(cache_log_path, slide_fullname, "OK - Copied")
            except Exception as e:
                error_msg = f"ERROR copying: {e}"
                update_log(cache_log_path, slide_fullname, error_msg)
                logger.error(f"Failed to copy {source_file_path} to cache: {e}")
                # Attempt cleanup, ignore errors
                try:
                    if os.path.exists(cache_file_path + ".lock"):
                        remove_lock(cache_file_path)
                    if os.path.exists(cache_file_path):  # Remove partially copied file
                        os.remove(cache_file_path)
                except OSError:
                    pass
            finally:
                if os.path.exists(cache_file_path + ".lock"):
                    try:
                        remove_lock(cache_file_path)
                    except OSError as lock_err:
                        logger.warning(
                            f"Could not remove lock file for {cache_file_path} after operation: {lock_err}"
                        )
        logger.info("Cache population finished.")

    def run_segmentation_job(
        self,
        segmentation_model: SegmentationModel,
        seg_mag: int = 10,
        holes_are_tissue: bool = False,
        batch_size: int = 16,
        artifact_remover_model: Optional[SegmentationModel] = None,
        device: str = "cuda:0",
    ) -> str:
        """
        Performs tissue segmentation on the targeted WSIs.

        Uses the provided `segmentation_model` to identify tissue regions.
        Optionally uses an `artifact_remover_model` for refinement. Saves results
        (thumbnails, contours, GeoJSON) in subdirectories under `job_dir`.

        Args:
            segmentation_model: A pre-trained PyTorch model for tissue segmentation.
            seg_mag: Target magnification (e.g., 10 for 10x) for segmentation. Defaults to 10.
            holes_are_tissue: If True, holes within tissue contours are considered tissue.
                              If False, they are excluded. Defaults to False.
            batch_size: Batch size for model inference during segmentation. Defaults to 16.
            artifact_remover_model: Optional second model to refine segmentation, often
                                    used to remove artifacts like pen marks. Defaults to None.
            device: The device for PyTorch computations (e.g., 'cuda:0', 'cpu'). Defaults to 'cuda:0'.

        Returns:
            Absolute path to the directory where GeoJSON contour files are saved.

        Raises:
            RuntimeError: If an error occurs during segmentation and `skip_errors` is False.
        """
        segmentation_job_name = "segmentation_results"
        paths = self._get_job_paths(
            segmentation_job_name, ["contours_geojson", "contours", "thumbnails"]
        )
        geojson_dir = paths["contours_geojson"]
        log_fp = paths["log"]

        # --- Save Configuration ---
        sig = signature(self.run_segmentation_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        # Add model names if available
        if hasattr(segmentation_model, "model_name"):
            local_attrs["segmentation_model_name"] = segmentation_model.model_name
        if artifact_remover_model and hasattr(artifact_remover_model, "model_name"):
            local_attrs["artifact_remover_model_name"] = (
                artifact_remover_model.model_name
            )
        self.save_config(
            saveto=paths["config"],
            local_attrs=local_attrs,
            ignore=["self", "segmentation_model", "artifact_remover_model"],
        )
        logger.info(f"Starting segmentation job. Results will be in {paths['base']}")

        progress_bar = tqdm(
            self.wsis, desc="Segmenting tissue", total=len(self.wsis), unit="slide"
        )

        for wsi in progress_bar:
            slide_fullname = wsi.name + wsi.ext
            geojson_path = os.path.join(geojson_dir, f"{wsi.name}.geojson")
            progress_bar.set_postfix_str(f"{slide_fullname}")

            # --- Pre-computation Checks ---
            if os.path.exists(geojson_path) and not is_locked(geojson_path):
                update_log(log_fp, slide_fullname, "DONE - Already segmented")
                self.cleanup_wsi_cache(slide_fullname)  # Clean up cache if done
                continue
            if is_locked(geojson_path):
                update_log(log_fp, slide_fullname, "SKIP - Locked")
                continue

            # Check if original WSI file exists at its load path (which could be cache)
            if not os.path.exists(wsi.slide_path):
                update_log(
                    log_fp,
                    slide_fullname,
                    f"SKIP - WSI file not found at {wsi.slide_path}",
                )
                continue

            # --- Perform Segmentation ---
            try:
                create_lock(geojson_path)
                update_log(log_fp, slide_fullname, "LOCK - Segmenting")
                wsi._lazy_initialize()  # Ensure WSI is loaded

                generated_geojson_path = wsi.segment_tissue(
                    segmentation_model=segmentation_model,
                    target_mag=seg_mag,
                    holes_are_tissue=holes_are_tissue,
                    job_dir=paths["base"],  # Pass base segmentation dir
                    batch_size=batch_size,
                    device=device,
                    verbose=False,
                )

                if artifact_remover_model is not None:
                    logger.info(f"Applying artifact remover to {slide_fullname}")
                    generated_geojson_path = wsi.segment_tissue(
                        segmentation_model=artifact_remover_model,
                        target_mag=getattr(
                            artifact_remover_model, "target_mag", seg_mag
                        ),
                        holes_are_tissue=False,  # Artifact remover usually removes holes (artifacts)
                        job_dir=paths["base"],
                        batch_size=batch_size,
                        device=device,
                        verbose=False,
                    )

                # Verify output
                if not os.path.exists(generated_geojson_path):
                    raise FileNotFoundError(
                        f"Segmentation output {generated_geojson_path} not created."
                    )
                try:
                    gdf = gpd.read_file(generated_geojson_path, rows=1)
                    status = "DONE - Segmented"
                    if gdf.empty:
                        status = "WARN - Empty GeoDataFrame"
                        logger.warning(
                            f"Empty segmentation result for {slide_fullname}"
                        )
                    update_log(log_fp, slide_fullname, status)
                except Exception as gdf_err:
                    update_log(
                        log_fp, slide_fullname, f"ERROR reading GeoJSON: {gdf_err}"
                    )
                    raise ValueError(
                        f"Could not read generated GeoJSON: {gdf_err}"
                    ) from gdf_err

            except Exception as e:
                error_msg = f"ERROR during segmentation: {e}"
                update_log(log_fp, slide_fullname, error_msg)
                logger.error(f"Error segmenting {slide_fullname}: {e}")
                if isinstance(e, KeyboardInterrupt):
                    print("Segmentation interrupted.")
                    raise e
                if not self.skip_errors:
                    raise RuntimeError(f"Error segmenting {slide_fullname}: {e}") from e
                # Continue loop if skipping errors
            finally:
                if os.path.exists(geojson_path + ".lock"):
                    try:
                        remove_lock(geojson_path)
                    except OSError as lock_err:
                        logger.warning(
                            f"Could not remove lock {geojson_path}.lock: {lock_err}"
                        )
                wsi.close()  # Close WSI handle to free resources
                self.cleanup_wsi_cache(slide_fullname)

        logger.info(f"Segmentation job finished. GeoJSONs in: {geojson_dir}")
        return geojson_dir

    def run_patching_job(
        self,
        target_magnification: int,
        patch_size: int,
        overlap: int = 0,
        patch_dir_name: Optional[str] = None,
        visualize: bool = True,
        min_tissue_proportion: float = 0.0,
    ) -> str:
        """Extracts patch coordinates from segmented tissue regions for each WSI."""
        if patch_dir_name is None:
            patch_dir_name = (
                f"patches_{target_magnification}x_{patch_size}px_{overlap}ovlp"
            )

        paths = self._get_job_paths(
            patch_dir_name, ["patches", "visualization"] if visualize else ["patches"]
        )
        coords_h5_dir = paths["patches"]  # HDF5 files go here
        viz_dir = paths.get("visualization")  # Will be None if visualize=False
        log_fp = paths["log"]

        # --- Save Configuration ---
        sig = signature(self.run_patching_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        self.save_config(
            saveto=paths["config"], local_attrs=local_attrs, ignore=["self"]
        )
        logger.info(f"Starting patching job. Results will be in {paths['base']}")

        progress_bar = tqdm(
            self.wsis,
            desc=f"Extracting patch coordinates ({patch_dir_name})",
            total=len(self.wsis),
            unit="slide",
        )

        for wsi in progress_bar:
            slide_fullname = wsi.name + wsi.ext
            coords_h5_path = os.path.join(coords_h5_dir, f"{wsi.name}_patches.h5")
            progress_bar.set_postfix_str(f"{slide_fullname}")

            # --- Pre-computation Checks ---
            if os.path.exists(coords_h5_path) and not is_locked(coords_h5_path):
                update_log(log_fp, slide_fullname, "DONE - Coords already generated")
                self.cleanup_wsi_cache(slide_fullname)  # Clean up cache if done
                continue
            if is_locked(coords_h5_path):
                update_log(log_fp, slide_fullname, "SKIP - Locked")
                continue

            # Check if original WSI file exists at its load path (which could be cache)
            if not os.path.exists(wsi.slide_path):
                update_log(
                    log_fp,
                    slide_fullname,
                    f"SKIP - WSI file not found at {wsi.slide_path}",
                )
                continue

            segmentation_path = wsi.tissue_seg_path  # Should be set if segmentation ran
            if segmentation_path is None or not os.path.exists(segmentation_path):
                update_log(
                    log_fp, slide_fullname, "SKIP - Segmentation GeoJSON not found"
                )
                continue
            try:  # Check if GeoJSON is empty
                gdf = gpd.read_file(segmentation_path, rows=1)
                if gdf.empty:
                    update_log(
                        log_fp,
                        slide_fullname,
                        "SKIP - Empty GeoDataFrame for segmentation",
                    )
                    continue
            except Exception as gdf_err:
                update_log(log_fp, slide_fullname, f"ERROR reading GeoJSON: {gdf_err}")
                if not self.skip_errors:
                    raise RuntimeError(
                        f"Error reading GeoJSON {segmentation_path}"
                    ) from gdf_err
                continue

            # --- Perform Patching ---
            try:
                create_lock(coords_h5_path)
                update_log(log_fp, slide_fullname, "LOCK - Generating coords")
                wsi._lazy_initialize()  # Ensure WSI loaded

                generated_coords_path = wsi.extract_tissue_coords(
                    target_mag=target_magnification,
                    patch_size=patch_size,
                    save_coords=paths["base"],  # Pass base dir for patching job
                    overlap=overlap,
                    min_tissue_proportion=min_tissue_proportion,
                )

                if not os.path.exists(generated_coords_path):
                    raise FileNotFoundError(
                        f"Coordinate file {generated_coords_path} not created."
                    )

                if viz_dir:
                    wsi.visualize_coords(
                        coords_path=generated_coords_path,
                        save_patch_viz=viz_dir,
                    )
                update_log(log_fp, slide_fullname, "DONE - Coords generated")

            except Exception as e:
                error_msg = f"ERROR during patching: {e}"
                update_log(log_fp, slide_fullname, error_msg)
                logger.error(f"Error patching {slide_fullname}: {e}")
                if isinstance(e, KeyboardInterrupt):
                    print("Patching interrupted.")
                    raise e
                if not self.skip_errors:
                    raise RuntimeError(f"Error patching {slide_fullname}: {e}") from e
            finally:
                if os.path.exists(coords_h5_path + ".lock"):
                    try:
                        remove_lock(coords_h5_path)
                    except OSError as lock_err:
                        logger.warning(
                            f"Could not remove lock {coords_h5_path}.lock: {lock_err}"
                        )
                wsi.close()  # Close WSI handle to free resources
                self.cleanup_wsi_cache(slide_fullname)

        logger.info(f"Patching job finished. Coordinates in: {coords_h5_dir}")
        return coords_h5_dir  # Return path to HDF5 coordinate files

    def run_patch_feature_extraction_job(
        self,
        coords_h5_dir: str,  # Dir containing HDF5 patch coord files
        patch_encoder: BasePatchEncoder,  # Use new base class for type hinting
        device: str = "cuda:0",
        saveas: str = "h5",
        batch_limit: int = 512,
        features_dir_name: Optional[str] = None,
    ) -> str:
        """Extracts patch-level features using a specified patch encoder model."""
        # --- Determine Paths ---
        if not os.path.isdir(coords_h5_dir):
            raise FileNotFoundError(f"Coordinates directory not found: {coords_h5_dir}")
        # Assume coords_h5_dir is like .../job_dir/patch_job_name/patches/
        patching_base_dir = os.path.dirname(coords_h5_dir)

        enc_name = getattr(patch_encoder, "enc_name", "custom_encoder")
        if features_dir_name is None:
            features_dir_name = f"features_{enc_name}"

        # Feature files will live alongside the 'patches' dir
        features_base_dir = os.path.join(patching_base_dir, features_dir_name)
        os.makedirs(features_base_dir, exist_ok=True)

        paths = {
            "base": features_base_dir,
            "config": os.path.join(features_base_dir, "_config_patch_features.json"),
            "log": os.path.join(features_base_dir, "_log_patch_features.txt"),
        }
        log_fp = paths["log"]

        # --- Save Configuration ---
        sig = signature(self.run_patch_feature_extraction_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        local_attrs["patch_encoder_name"] = enc_name
        local_attrs["patch_encoder_embedding_dim"] = getattr(
            patch_encoder, "embedding_dim", "unknown"
        )

        self.save_config(
            saveto=paths["config"],
            local_attrs=local_attrs,
            ignore=["self", "patch_encoder"],
        )
        logger.info(
            f"Starting patch feature extraction ({features_dir_name}). Results in {paths['base']}"
        )

        progress_bar = tqdm(
            self.wsis,
            desc=f"Extracting patch features ({features_dir_name})",
            total=len(self.wsis),
            unit="slide",
        )

        for wsi in progress_bar:
            slide_fullname = wsi.name + wsi.ext
            coord_h5_path = os.path.join(coords_h5_dir, f"{wsi.name}_patches.h5")
            feature_file_path = os.path.join(features_base_dir, f"{wsi.name}.{saveas}")
            progress_bar.set_postfix_str(f"{slide_fullname}")

            # --- Pre-computation Checks ---
            if os.path.exists(feature_file_path) and not is_locked(feature_file_path):
                update_log(log_fp, slide_fullname, "DONE - Features already extracted")
                self.cleanup_wsi_cache(slide_fullname)  # Clean up cache if done
                continue
            if is_locked(feature_file_path):
                update_log(log_fp, slide_fullname, "SKIP - Locked")
                continue

            # Check if original WSI file exists at its load path (which could be cache)
            if not os.path.exists(wsi.slide_path):
                update_log(
                    log_fp,
                    slide_fullname,
                    f"SKIP - WSI file not found at {wsi.slide_path}",
                )
                continue

            if not os.path.exists(coord_h5_path):
                update_log(
                    log_fp,
                    slide_fullname,
                    f"SKIP - Coordinate file not found: {coord_h5_path}",
                )
                continue

            # --- Perform Feature Extraction ---
            try:
                create_lock(feature_file_path)
                update_log(log_fp, slide_fullname, "LOCK - Extracting patch features")
                wsi._lazy_initialize()  # Ensure WSI loaded

                generated_feature_path = wsi.extract_patch_features(
                    patch_encoder=patch_encoder,
                    coords_path=coord_h5_path,
                    save_features=features_base_dir,  # Pass the target directory
                    device=device,
                    saveas=saveas,
                    batch_limit=batch_limit,
                )

                if not os.path.exists(generated_feature_path):
                    raise FileNotFoundError(
                        f"Feature file {generated_feature_path} not created."
                    )
                update_log(log_fp, slide_fullname, "DONE - Features extracted")

            except Exception as e:
                error_msg = f"ERROR during patch feature extraction: {e}"
                update_log(log_fp, slide_fullname, error_msg)
                logger.error(
                    f"Error extracting patch features for {slide_fullname}: {e}"
                )
                if isinstance(e, KeyboardInterrupt):
                    print("Patch feature extraction interrupted.")
                    raise e
                if not self.skip_errors:
                    raise RuntimeError(
                        f"Error extracting patch features for {slide_fullname}: {e}"
                    ) from e
            finally:
                if os.path.exists(feature_file_path + ".lock"):
                    try:
                        remove_lock(feature_file_path)
                    except OSError as lock_err:
                        logger.warning(
                            f"Could not remove lock {feature_file_path}.lock: {lock_err}"
                        )
                wsi.close()  # Close WSI handle to free resources
                self.cleanup_wsi_cache(slide_fullname)

        logger.info(
            f"Patch feature extraction finished. Features in: {features_base_dir}"
        )
        return features_base_dir

    def run_slide_feature_extraction_job(
        self,
        slide_encoder: BaseSlideEncoder,  # Use new base class for type hinting
        coords_h5_dir: str,  # Base directory for the patching job (contains 'patches' and 'features_...')
        device: str = "cuda:0",
        saveas: str = "h5",
        batch_limit_for_patch_features: int = 512,  # Used if auto-generating patch features
        slide_features_dir_name: Optional[str] = None,
    ) -> str:
        """Extracts slide-level features using a specified slide encoder model."""
        # --- Determine Paths and Required Patch Encoder ---
        if not os.path.isdir(coords_h5_dir):
            raise FileNotFoundError(
                f"Coordinates directory (e.g., .../job_dir/patch_job_name) not found: {coords_h5_dir}"
            )

        slide_enc_name = getattr(slide_encoder, "enc_name", "custom_slide_encoder")
        required_patch_enc_name = None
        # Infer required patch encoder name from slide encoder name
        if slide_enc_name.startswith("mean-"):
            # For mean-pooled encoders, the patch encoder is the suffix
            required_patch_enc_name = slide_enc_name.split("mean-", 1)[1]
        elif slide_enc_name in slide_to_patch_encoder_name:
            # For specific slide encoders, use the predefined mapping
            required_patch_enc_name = slide_to_patch_encoder_name[slide_enc_name]

        # Bug Fix: Use this consistent expected_patch_features_dir for all related paths
        # This is the path where the patch features *should* be, and where they will be generated if missing.
        if (
            required_patch_enc_name
        ):  # Only define if a patch encoder is actually required
            expected_patch_features_dir = os.path.join(
                coords_h5_dir, f"features_{required_patch_enc_name}"
            )
        else:  # If no specific patch encoder is required (e.g., custom model), fallback or raise error
            # For models that don't need a specific patch encoder (e.g., direct image input, though not typical for SlideEncoders here)
            # or if the user passed a `patch_features_dir` directly which is not derived from a known encoder name
            # In this case, `patch_features_dir` should be treated as the direct input for patch features.
            # We assume for this method that `patch_features_dir` (argument) *is* the directory containing features.
            # If `required_patch_enc_name` is None, then `patch_features_dir` (argument) must be the actual path.
            # However, the method signature takes `coords_h5_dir` as the base patching job directory.
            # To be robust, if `required_patch_enc_name` is None, we need an explicit `patch_features_input_dir` argument.
            # For now, let's assume `slide_encoder` implies a `required_patch_enc_name` for this pipeline.
            # If a model needs different input (e.g. raw images, or different feature structure), it should have its own method.
            raise ValueError(
                f"Slide encoder '{slide_enc_name}' does not have a mapped patch encoder. Cannot proceed with auto-generation or locating patch features."
            )

        # Determine output directory for slide features
        if slide_features_dir_name is None:
            slide_features_dir_name = f"slide_features_{slide_enc_name}"
        slide_features_base_dir = os.path.join(
            coords_h5_dir,
            slide_features_dir_name,  # Slide features are children of the same patching job directory
        )
        os.makedirs(slide_features_base_dir, exist_ok=True)

        paths = {
            "base": slide_features_base_dir,
            "config": os.path.join(
                slide_features_base_dir, "_config_slide_features.json"
            ),
            "log": os.path.join(slide_features_base_dir, "_log_slide_features.txt"),
        }
        log_fp = paths["log"]

        # --- Auto-generate Patch Features if Missing ---
        if required_patch_enc_name:
            # Check if all patch feature files for this encoder exist
            all_patch_features_exist = all(
                os.path.exists(
                    os.path.join(expected_patch_features_dir, f"{wsi.name}.h5")
                )
                for wsi in self.wsis
            )

            if not all_patch_features_exist:
                logger.warning(
                    f"Required patch features ('{required_patch_enc_name}') missing in '{expected_patch_features_dir}'. Attempting generation."
                )
                try:
                    patch_encoder = patch_encoder_factory(required_patch_enc_name)
                    # The actual coordinate H5 files are in a 'patches' subfolder within coords_h5_dir
                    patch_coords_h5_dir_for_generation = os.path.join(
                        coords_h5_dir, "patches"
                    )

                    if not os.path.isdir(patch_coords_h5_dir_for_generation):
                        raise FileNotFoundError(
                            f"Coordinate directory '{patch_coords_h5_dir_for_generation}' needed for auto-generation not found."
                        )

                    # Run the patch feature job, ensuring it saves to the correct directory (`expected_patch_features_dir`)
                    self.run_patch_feature_extraction_job(
                        coords_h5_dir=patch_coords_h5_dir_for_generation,  # This is the path to the 'patches' dir
                        patch_encoder=patch_encoder,
                        device=device,
                        saveas="h5",  # Must be h5 for slide feature extraction
                        batch_limit=batch_limit_for_patch_features,
                        features_dir_name=os.path.basename(
                            expected_patch_features_dir
                        ),  # Ensure it saves to the desired 'features_...' folder
                    )
                    logger.info(
                        f"Auto-generation of patch features ({required_patch_enc_name}) complete."
                    )
                except Exception as patch_gen_e:
                    raise RuntimeError(
                        f"Failed to auto-generate required patch features ('{required_patch_enc_name}'). "
                        f"Generate manually or ensure they exist in '{expected_patch_features_dir}'. Error: {patch_gen_e}"
                    ) from patch_gen_e
        # No else needed: If required_patch_enc_name is None, it would have raised an error above.

        # --- Save Configuration ---
        sig = signature(self.run_slide_feature_extraction_job)
        local_attrs = {k: v for k, v in locals().items() if k in sig.parameters}
        local_attrs["slide_encoder_name"] = slide_enc_name
        local_attrs["slide_encoder_embedding_dim"] = getattr(
            slide_encoder, "embedding_dim", "unknown"
        )
        if required_patch_enc_name:
            local_attrs["required_patch_encoder"] = required_patch_enc_name

        self.save_config(
            saveto=paths["config"],
            local_attrs=local_attrs,
            ignore=["self", "slide_encoder"],
        )
        logger.info(
            f"Starting slide feature extraction ({slide_features_dir_name}). Results in {paths['base']}"
        )

        progress_bar = tqdm(
            self.wsis,
            desc=f"Extracting slide features ({slide_features_dir_name})",
            total=len(self.wsis),
            unit="slide",
        )

        for wsi in progress_bar:
            slide_fullname = wsi.name + wsi.ext
            # Bug Fix applied here: Consistently use `expected_patch_features_dir` for input path
            # to ensure we look where the features were (or should have been) generated.
            patch_feature_h5_path = os.path.join(
                expected_patch_features_dir, f"{wsi.name}.h5"
            )  # Assumes h5 input for features
            slide_feature_file_path = os.path.join(
                slide_features_base_dir, f"{wsi.name}.{saveas}"
            )
            progress_bar.set_postfix_str(f"{slide_fullname}")

            # --- Pre-computation Checks ---
            if os.path.exists(slide_feature_file_path) and not is_locked(
                slide_feature_file_path
            ):
                update_log(
                    log_fp, slide_fullname, "DONE - Slide features already extracted"
                )
                self.cleanup_wsi_cache(slide_fullname)  # Clean up cache if done
                continue
            if is_locked(slide_feature_file_path):
                update_log(log_fp, slide_fullname, "SKIP - Locked")
                continue
            if not os.path.exists(patch_feature_h5_path):
                # This should ideally not happen if auto-generation ran successfully, but good for robustness
                update_log(
                    log_fp,
                    slide_fullname,
                    f"SKIP - Required patch feature file not found: {patch_feature_h5_path}",
                )
                continue

            # Check if original WSI file exists at its load path (which could be cache)
            if not os.path.exists(wsi.slide_path):
                logger.debug(
                    f"WSI file {wsi.slide_path} missing, but proceeding with feature file (if not strictly needed)."
                )
                # update_log(log_fp, slide_fullname, f"INFO - WSI file missing at {wsi.slide_path}") # Optional logging

            # --- Perform Slide Feature Extraction ---
            try:
                create_lock(slide_feature_file_path)
                update_log(log_fp, slide_fullname, "LOCK - Extracting slide features")
                # Initialize WSI mainly for metadata access if needed by slide encoder (e.g., GigaPath needs patch_size_level0 from coords.attrs)
                wsi._lazy_initialize()

                generated_slide_feature_path = wsi.extract_slide_features(
                    patch_features_path=patch_feature_h5_path,
                    slide_encoder=slide_encoder,
                    save_features=slide_features_base_dir,  # Pass directory
                    device=device,
                    # saveas='h5' is implicit in wsi.extract_slide_features currently
                )

                if not os.path.exists(generated_slide_feature_path):
                    raise FileNotFoundError(
                        f"Slide feature file {generated_slide_feature_path} not created."
                    )
                update_log(log_fp, slide_fullname, "DONE - Slide features extracted")

            except Exception as e:
                error_msg = f"ERROR during slide feature extraction: {e}"
                update_log(log_fp, slide_fullname, error_msg)
                logger.error(
                    f"Error extracting slide features for {slide_fullname}: {e}"
                )
                if isinstance(e, KeyboardInterrupt):
                    print("Slide feature extraction interrupted.")
                    raise e
                if not self.skip_errors:
                    raise RuntimeError(
                        f"Error extracting slide features for {slide_fullname}: {e}"
                    ) from e
            finally:
                if os.path.exists(slide_feature_file_path + ".lock"):
                    try:
                        remove_lock(slide_feature_file_path)
                    except OSError as lock_err:
                        logger.warning(
                            f"Could not remove lock {slide_feature_file_path}.lock: {lock_err}"
                        )
                wsi.close()  # Close WSI handle to free resources
                self.cleanup_wsi_cache(slide_fullname)

        logger.info(
            f"Slide feature extraction finished. Features in: {slide_features_base_dir}"
        )
        return slide_features_base_dir

    def cleanup_wsi_cache(self, filename: str) -> None:
        """Removes the specified WSI file and its associated .mrxs directory from the cache if enabled."""
        if self.wsi_cache and self.clear_cache:
            cache_file_path = os.path.join(self.wsi_cache, filename)

            # Check for .mrxs subdirectory
            mrxs_dir_path = os.path.splitext(cache_file_path)[0]

            if os.path.exists(cache_file_path):
                if not is_locked(
                    cache_file_path
                ):  # Only clean if not locked by another process
                    try:
                        os.remove(cache_file_path)
                        logger.debug(f"Cleaned {filename} from cache.")
                        # Check for and remove associated .mrxs directory
                        if os.path.isdir(mrxs_dir_path) and filename.lower().endswith(
                            ".mrxs"
                        ):
                            shutil.rmtree(mrxs_dir_path)
                            logger.debug(
                                f"Cleaned associated .mrxs directory {os.path.basename(mrxs_dir_path)} from cache."
                            )
                    except OSError as e:
                        logger.warning(
                            f"Failed to remove {cache_file_path} from cache: {e}"
                        )
                # else: Logged by previous checks

    def save_config(
        self,
        saveto: PathLike,
        local_attrs: Optional[Dict[str, Any]] = None,
        ignore: Optional[List[str]] = None,
    ) -> None:
        """Saves the processor's configuration and job parameters to a JSON file."""
        if ignore is None:
            ignore = [
                "wsis",
                "loop",  # tqdm loop object
                "wsi_source",
                "wsi_cache",
            ]  # Exclude sensitive/large/redundant

        config_to_save = {}

        # Add instance attributes (filter sensitive/large ones)
        for k, v in vars(self).items():
            if k not in ignore and not k.startswith("_"):  # Exclude private attrs
                # Special handling for PathLike objects or objects that might be Path objects
                if isinstance(v, (os.PathLike, Path)):
                    v = str(v)
                try:
                    # Attempt to dump to ensure serializability with custom JSONsaver
                    # This is just a test; the actual dump uses the same logic
                    JSONsaver().encode(v)
                    config_to_save[k] = v
                except (TypeError, OverflowError):
                    config_to_save[k] = (
                        f"<Object type: {type(v).__name__}>"  # Represent non-serializable
                    )

        # Add/overwrite with local attributes from the specific job method
        if local_attrs:
            for k, v in local_attrs.items():
                if k not in ignore:
                    # Special handling for PathLike objects or objects that might be Path objects
                    if isinstance(v, (os.PathLike, Path)):
                        v = str(v)
                    try:
                        # Attempt to dump to ensure serializability with custom JSONsaver
                        JSONsaver().encode(v)
                        config_to_save[k] = v
                    except (TypeError, OverflowError):
                        # Special handling for models - just save name if possible
                        if isinstance(v, torch.nn.Module) and hasattr(v, "enc_name"):
                            config_to_save[k] = (
                                f"<Model: {getattr(v, 'enc_name', type(v).__name__)}>"
                            )
                        elif isinstance(v, torch.nn.Module) and hasattr(
                            v, "model_name"
                        ):
                            config_to_save[k] = (
                                f"<Model: {getattr(v, 'model_name', type(v).__name__)}>"
                            )
                        else:
                            config_to_save[k] = f"<Object type: {type(v).__name__}>"

        # Ensure directory exists and save
        try:
            os.makedirs(os.path.dirname(saveto), exist_ok=True)
            with open(saveto, "w") as f:
                # Use JSONsaver for the actual dump
                json.dump(
                    config_to_save, f, indent=4, cls=JSONsaver, ensure_ascii=False
                )
            logger.debug(f"Configuration saved successfully to {saveto}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {saveto}: {e}")

    def release(self) -> None:
        """
        Release all resources tied to the WSIs held by this Processor instance.
        Frees memory, closes file handles, and clears GPU memory.
        Should be called after processing is complete to avoid memory leaks.
        """
        if hasattr(self, "wsis"):
            for wsi in self.wsis:
                try:
                    wsi.close()  # Calls the backend-specific close method
                except Exception:
                    pass
            self.wsis.clear()

        # Also clear loop references (e.g., tqdm)
        if hasattr(self, "loop"):
            self.loop = None

        # Explicit garbage collection and CUDA cache release
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
