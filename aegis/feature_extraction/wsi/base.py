# aegis/feature_extraction/wsi/base.py
# Corrected code for segment_tissue method

from __future__ import annotations

import os
import warnings
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from aegis.feature_extraction.models.patch_encoders.factory import (
    BasePatchEncoder,  # For type hinting
)

# Import segmentation models and patch/slide encoders for type hinting and internal use
from aegis.feature_extraction.models.segmentation.factory import (
    SegmentationModel,  # For type hinting
)
from aegis.feature_extraction.models.slide_encoders.factory import (
    BaseSlideEncoder,  # For type hinting
)
from aegis.feature_extraction.utils.io import (
    get_num_workers,
    mask_to_gdf,
    overlay_gdf_on_thumbnail,
    read_coords,
    read_coords_legacy,
    save_h5,
)
from aegis.feature_extraction.wsi.dataset import (
    WSIPatcherDataset,  # Use WSIPatcherDataset for dataloader
)
from aegis.feature_extraction.wsi.patching import (
    WSIPatcher,  # Use WSIPatcher for type hinting
)

ReadMode = Literal["pil", "numpy"]


class WSI:
    """
    The `WSI` class provides an interface to work with Whole Slide Images (WSIs).
    It supports lazy initialization, metadata extraction, tissue segmentation,
    patching, and feature extraction. The class handles various WSI file formats and
    offers utilities for integration with AI models.

    Attributes
    ----------
    slide_path : str
        Path to the WSI file (the path from which the WSI object is currently loaded).
    original_path : str
        The original path to the WSI file, typically in the source directory. Useful for caching logic.
    name : str
        Name of the WSI (inferred from the file path if not provided).
    custom_mpp_keys : List[str]
        Custom keys for extracting microns per pixel (MPP) and magnification metadata.
    lazy_init : bool
        Indicates whether lazy initialization is used.
    tissue_seg_path : str
        Path to a tissue segmentation mask (if available).
    width : int
        Width of the WSI in pixels (set during lazy initialization).
    height : int
        Height of the WSI in pixels (set during lazy initialization).
    dimensions : Tuple[int, int]
        (width, height) tuple of the WSI (set during lazy initialization).
    mpp : float
        Microns per pixel (set during lazy initialization or inferred).
    mag : float
        Estimated magnification level (set during lazy initialization or inferred).
    level_count : int
        Number of resolution levels in the WSI (set during lazy initialization).
    level_downsamples : List[float]
        Downsampling factors for each pyramid level (set during lazy initialization).
    level_dimensions : List[Tuple[int, int]]
        Dimensions of the WSI at each pyramid level (set during lazy initialization).
    properties : dict
        Metadata properties extracted from the image backend (set during lazy initialization).
    img : Any
        Backend-specific image object used for reading regions (set during lazy initialization).
    gdf_contours : geopandas.GeoDataFrame
        Tissue segmentation mask as a GeoDataFrame, if available (set during lazy initialization).
    """

    def __init__(
        self,
        slide_path: str,  # Path where the WSI is *currently* located (could be cache)
        original_path: Optional[str] = None,  # Original path (source directory)
        name: Optional[str] = None,
        tissue_seg_path: Optional[str] = None,
        custom_mpp_keys: Optional[List[str]] = None,
        lazy_init: bool = True,
        mpp: Optional[float] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the `WSI` object for working with a Whole Slide Image (WSI).

        Args:
        -----
        slide_path : str
            Path to the WSI file (where it is currently accessible, e.g., in cache).
        original_path : str, optional
            The original full path to the WSI file in its source directory. Used for tracking
            and potentially for caching logic. If None, defaults to `slide_path`.
        name : str, optional
            Optional name for the WSI. Defaults to the filename (without extension).
        tissue_seg_path : str, optional
            Path to the tissue segmentation mask file. Defaults to None.
        custom_mpp_keys : Optional[List[str]]
            Custom keys for extracting MPP and magnification metadata. Defaults to None.
        lazy_init : bool, optional
            If True, defer loading the WSI until required. Defaults to True.
        mpp: float, optional
            If not None, will be the reference micron per pixel (mpp). Handy when mpp is not provided in the WSI.
        max_workers (Optional[int]): Maximum number of workers for data loading

        """
        self.slide_path = slide_path
        self.original_path = original_path if original_path is not None else slide_path
        if name is None:
            self.name, self.ext = os.path.splitext(os.path.basename(slide_path))
        else:
            self.name, self.ext = os.path.splitext(name)
        self.tissue_seg_path = tissue_seg_path
        self.custom_mpp_keys = custom_mpp_keys

        self.width, self.height = None, None  # Placeholder dimensions
        self.mpp = mpp  # Placeholder microns per pixel. Defaults will be None unless specified in constructor.
        self.mag = None  # Placeholder magnification
        self._is_initialized = (
            False  # Internal flag to track if _lazy_initialize has run
        )
        self.max_workers = max_workers

        if not lazy_init:  # User explicitly requested immediate initialization
            self._lazy_initialize()
        else:  # Lazy init is True, so only attempt to load existing contours for quick check
            if self.tissue_seg_path is not None and os.path.exists(
                self.tissue_seg_path
            ):
                try:
                    self.gdf_contours = gpd.read_file(self.tissue_seg_path)
                except Exception as e:
                    warnings.warn(
                        f"Failed to load existing GeoJSON for {self.name}: {e}. Will re-segment if needed."
                    )
                    self.gdf_contours = None
                    self.tissue_seg_path = None  # Clear path if loading failed.
            else:
                self.gdf_contours = None
                self.tissue_seg_path = None  # Ensure it's None if file doesn't exist

    @abstractmethod
    def _lazy_initialize(self) -> None:
        """Abstract method to initialize the WSI object."""
        pass

    def create_patcher(
        self,
        patch_size: int,
        src_pixel_size: Optional[float] = None,
        dst_pixel_size: Optional[float] = None,
        src_mag: Optional[int] = None,
        dst_mag: Optional[int] = None,
        overlap: int = 0,
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False,
        custom_coords: Optional[np.ndarray] = None,
        min_tissue_proportion: float = 0.0,  # Changed from 'threshold'
        pil: bool = False,
    ) -> WSIPatcher:
        """
        Create a patcher object for extracting patches from the WSI.

        Args:
        -----
        patch_size : int
            Size of each patch in pixels for the output (at `dst_mag` or `dst_pixel_size`).
        src_pixel_size : float, optional
            Pixel size in um/px of the source WSI. If None, `self.mpp` is used.
        dst_pixel_size : float, optional
            Desired pixel size in um/px for the output patches. If None, `dst_mag` must be provided.
        src_mag : int, optional
            Magnification of the source WSI. If None, `self.mag` is used.
        dst_mag : int, optional
            Desired magnification for the output patches. If None, `dst_pixel_size` must be provided.
        overlap (int, optional): Overlap between patches in pixels (at the output resolution). Defaults to 0.
        mask (gpd.GeoDataFrame, optional): geopandas dataframe of Polygons to filter patches. Defaults to None.
        coords_only (bool, optional): If True, the patcher yields only (x, y) coordinates instead of (patch, x, y). Default to False.
        custom_coords (np.ndarray, optional): Pre-defined (N, 2) array of (x, y) coordinates (at level 0) to extract. If provided, `mask`, grid generation, and `overlap` are ignored. Defaults to None.
        min_tissue_proportion: float, optional
            Minimum proportion of the patch area (at `dst_mag`/`dst_pixel_size`) that must be tissue to be kept.
            Applied only if `mask` is provided. Between 0.0 and 1.0. Defaults to 0.0 (any tissue presence).
        pil (bool, optional): If True, patches are returned as `PIL.Image` objects. Otherwise, as NumPy arrays. Defaults to False.

        Returns:
        --------
        WSIPatcher:
            An object for extracting patches.
        """
        # Ensure WSI is initialized to access metadata like self.mpp, self.mag
        self._lazy_initialize()

        # Resolve src_pixel_size and src_mag if not provided
        if src_pixel_size is None:
            if self.mpp is None:
                raise ValueError(
                    "src_pixel_size or self.mpp must be available to create patcher."
                )
            src_pixel_size = self.mpp

        if src_mag is None:
            if self.mag is None:
                raise ValueError(
                    "src_mag or self.mag must be available to create patcher."
                )
            src_mag = self.mag

        # Determine which patcher to use based on WSI backend
        if "openslide" in str(type(self)).lower():
            from aegis.feature_extraction.wsi.patching import OpenSlideWSIPatcher

            PatcherClass = OpenSlideWSIPatcher
        elif "cucim" in str(type(self)).lower():
            from aegis.feature_extraction.wsi.patching import CuImageWSIPatcher

            PatcherClass = CuImageWSIPatcher
        elif "image" in str(type(self)).lower():  # Assuming a PIL-based ImageWSI
            from aegis.feature_extraction.wsi.patching import (
                NumpyWSIPatcher,
            )  # Or a more specific one if available

            PatcherClass = NumpyWSIPatcher
        else:
            # Fallback or error
            raise TypeError(f"No specific patcher found for WSI type {type(self)}.")

        return PatcherClass(
            wsi=self,  # Pass self (WSI instance)
            patch_size=patch_size,
            src_pixel_size=src_pixel_size,
            dst_pixel_size=dst_pixel_size,
            src_mag=src_mag,
            dst_mag=dst_mag,
            overlap=overlap,
            mask=mask,
            coords_only=coords_only,
            custom_coords=custom_coords,
            min_tissue_proportion=min_tissue_proportion,
            pil=pil,
        )

    def _fetch_magnification(
        self, custom_mpp_keys: Optional[List[str]] = None
    ) -> Optional[int]:
        """
        The `_fetch_magnification` function of the class `WSI` calculates the magnification level
        of the WSI based on the microns per pixel (MPP) value or other metadata. The magnification levels are
        approximated to commonly used values such as 80x, 40x, 20x, etc. If the MPP is unavailable or insufficient
        for calculation, it attempts to fallback to metadata-based values.

        Args:
        -----
        custom_mpp_keys : Optional[List[str]], optional
            Custom keys to search for MPP values in the WSI properties. Defaults to None.

        Returns:
        --------
        Optional[int]]:
            The approximated magnification level, or None if the magnification could not be determined.

        Raises:
        -------
        ValueError:
            If the identified MPP is too low for valid magnification values.
        """
        if (
            self.mpp is None
        ):  # Attempt to get MPP from backend if not set by constructor
            try:
                mpp_x = self._fetch_mpp(custom_mpp_keys)
            except (
                ValueError
            ):  # If backend cannot provide MPP, then cannot infer magnification from MPP
                mpp_x = None
        else:  # MPP already provided in constructor
            mpp_x = self.mpp

        if mpp_x is not None:
            # Approximate standard magnifications based on common MPP values (e.g., 40x is ~0.25 MPP)
            # These are inverse relationships: higher mag -> lower mpp
            if mpp_x < 0.16:
                return 80
            elif mpp_x < 0.2:
                return 60
            elif mpp_x < 0.3:
                return 40
            elif mpp_x < 0.6:
                return 20
            elif mpp_x < 1.2:
                return 10
            elif mpp_x < 2.4:
                return 5
            else:
                warnings.warn(
                    f"Identified mpp ({mpp_x}) is unusually high. It may indicate a very low magnification or an error in MPP metadata."
                )
                return 1  # Fallback to 1x or lowest common mag

        return None  # Could not infer magnification from MPP

    @torch.inference_mode()
    @torch.autocast(
        device_type="cuda", dtype=torch.float16
    )  # Enable autocast for faster inference
    def segment_tissue(
        self,
        segmentation_model: SegmentationModel,
        target_mag: int = 10,
        holes_are_tissue: bool = True,
        job_dir: Optional[str] = None,
        batch_size: int = 16,
        device: str = "cuda:0",
        verbose: bool = False,
    ) -> str:
        """
        The `segment_tissue` function segments tissue regions in the WSI using
        a specified segmentation model. It processes the WSI at a target magnification level, optionally
        treating holes in the mask as tissue. The segmented regions are saved as thumbnails and GeoJSON contours.

        Args:
        -----
        segmentation_model : SegmentationModel (torch.nn.Module)
            The model used for tissue segmentation. Must be an instance of `SegmentationModel`
            (from `aegis.feature_extraction.models.segmentation.factory`).
        target_mag : int, optional
            Target magnification level for segmentation. Defaults to 10.
        holes_are_tissue : bool, optional
            Specifies whether to treat holes within tissue regions as part of the tissue. Defaults to True.
        job_dir :  Optional[str], optional
            Directory to save the segmentation results. Required.
        batch_size : int, optional
            Batch size for processing patches. Defaults to 16.
        device (str):
            The computation device to use (e.g., 'cuda:0' for GPU or 'cpu' for CPU).
        verbose: bool, optional:
            Whether to print segmentation progress bar. Defaults to False.

        Returns:
        --------
        str:
            The absolute path to where the segmentation as GeoJSON is saved.

        Raises:
        -------
        ValueError: If `job_dir` is None.
        """

        if job_dir is None:
            raise ValueError("`job_dir` must be provided to save segmentation results.")

        self._lazy_initialize()  # Ensure WSI object is fully loaded and self.width/height populated

        # --- BUG FIX START ---
        # These lines were being executed before _lazy_initialize() in the subclass
        # had populated self.width and self.height.
        # Moving them *after* _lazy_initialize() ensures dimensions are available.
        max_dimension = 1000
        # Check if width/height are still None after _lazy_initialize() (which implies an error during init)
        if self.width is None or self.height is None:
            raise RuntimeError(
                f"WSI dimensions (width/height) are None after initialization for {self.name}. Cannot perform segmentation."
            )

        # Debug print (can be removed in production)
        print(f"self.width: {self.width}, self.height: {self.height}")

        if self.width > self.height:
            thumbnail_width = max_dimension
            thumbnail_height = int(thumbnail_width * self.height / self.width)
        else:
            thumbnail_height = max_dimension
            thumbnail_width = int(thumbnail_height * self.width / self.height)
        thumbnail_pil = self.get_thumbnail((thumbnail_width, thumbnail_height))
        thumbnail_np = np.array(thumbnail_pil)  # Convert to numpy for OpenCV
        # --- BUG FIX END ---

        # Move model to device and set to eval mode
        segmentation_model.to(device)
        segmentation_model.eval()

        # Special handling for classic segmenters like CLAM that operate on whole images
        if type(segmentation_model).__name__.lower() in [
            "clamsegmenter",
            "classicsegmenter",
        ]:
            from torchvision import transforms

            # These models expect a single downsampled image.
            # We'll create a thumbnail at a specific magnification suitable for them.
            seg_mag = (
                1.25  # A low magnification like 1.25x is typical for classic methods
            )
            seg_mpp = 10 / seg_mag
            mpp_reduction_factor = self.mpp / seg_mpp

            thumb_w = int(self.width / (self.mpp / seg_mpp))
            thumb_h = int(self.height / (self.mpp / seg_mpp))

            image_for_seg = self.get_thumbnail((thumb_w, thumb_h))
            image_np = np.array(image_for_seg)

            # The ClamSegmenter's forward method expects a tensor
            img_tensor = transforms.ToTensor()(image_np).unsqueeze(0).to(device)

            # The forward pass returns a binary mask tensor
            binary_mask_tensor = segmentation_model(img_tensor)

            # Convert to numpy, remove batch dim, and scale to 255 for contour finding
            binary_mask_for_contours = (
                binary_mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8) * 255
            )

        else:
            # Get patch iterator for the segmentation model's target input resolution
            destination_mpp = (
                10 / segmentation_model.target_mag
            )  # MPP at model's target magnification

            if self.mpp is None:
                raise ValueError(
                    f"WSI {self.name} does not have MPP information. Cannot segment without it."
                )

            # Create patcher for segmentation
            patcher = self.create_patcher(
                patch_size=segmentation_model.input_size,
                src_pixel_size=self.mpp,
                dst_pixel_size=destination_mpp,
                mask=None,
                min_tissue_proportion=0.0,
                pil=True,
            )

            precision = segmentation_model.precision
            eval_transforms = segmentation_model.eval_transforms
            dataset = WSIPatcherDataset(patcher, eval_transforms)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=get_num_workers(batch_size, max_workers=self.max_workers),
                pin_memory=True,
            )

            mpp_reduction_factor = self.mpp / destination_mpp
            mask_width_at_target_mpp, mask_height_at_target_mpp = (
                int(round(self.width * mpp_reduction_factor)),
                int(round(self.height * mpp_reduction_factor)),
            )

            stitched_mask = np.zeros(
                (mask_height_at_target_mpp, mask_width_at_target_mpp), dtype=np.uint8
            )

            dataloader = tqdm(
                dataloader, desc=f"Segmenting {self.name}", disable=not verbose
            )

            for imgs, (xcoords_level0, ycoords_level0) in dataloader:
                imgs = imgs.to(device, dtype=precision)

                with torch.autocast(
                    device_type=device.split(":")[0],
                    dtype=precision,
                    enabled=(precision != torch.float32),
                ):
                    preds = segmentation_model(imgs).cpu().numpy()

                x_starts_target_mpp = np.clip(
                    np.round(xcoords_level0.numpy() * mpp_reduction_factor).astype(int),
                    0,
                    mask_width_at_target_mpp - 1,
                )
                y_starts_target_mpp = np.clip(
                    np.round(ycoords_level0.numpy() * mpp_reduction_factor).astype(int),
                    0,
                    mask_height_at_target_mpp - 1,
                )

                patch_input_size = segmentation_model.input_size

                for i in range(len(preds)):
                    x_start, y_start = x_starts_target_mpp[i], y_starts_target_mpp[i]
                    x_end = min(x_start + patch_input_size, mask_width_at_target_mpp)
                    y_end = min(y_start + patch_input_size, mask_height_at_target_mpp)

                    if x_start >= x_end or y_start >= y_end:
                        continue

                    patch_pred = preds[i][: y_end - y_start, : x_end - x_start]
                    stitched_mask[y_start:y_end, x_start:x_end] += patch_pred

            binary_mask_for_contours = (stitched_mask > 0).astype(np.uint8) * 255

        # Define save paths
        thumbnail_saveto = os.path.join(job_dir, "thumbnails", f"{self.name}.jpg")
        os.makedirs(os.path.dirname(thumbnail_saveto), exist_ok=True)
        thumbnail_pil.save(thumbnail_saveto)  # Save the original thumbnail

        gdf_saveto = os.path.join(job_dir, "contours_geojson", f"{self.name}.geojson")
        os.makedirs(os.path.dirname(gdf_saveto), exist_ok=True)

        # Convert the binary mask (at `target_mag` resolution) to GeoDataFrame contours
        # `pixel_size` passed to `mask_to_gdf` is the original WSI's MPP for area calculations.
        # `contour_scale` is used to scale the contours back to level 0 coordinates if the mask was generated at a lower resolution than level 0.
        # Here, `contour_scale` is 1 / mpp_reduction_factor, converting coordinates from the `target_mag` resolution back to level 0 pixels.
        gdf_contours = mask_to_gdf(
            mask=binary_mask_for_contours,
            max_nb_holes=(
                0
                if holes_are_tissue
                else getattr(segmentation_model, "max_holes_to_fill", 20)
            ),  # Use model's attribute or default
            min_contour_area=1000,  # Default: 1000. Area in pixels at the mask generation resolution
            pixel_size=self.mpp,  # Use original WSI MPP for consistency of area units (um^2)
            contour_scale=1
            / mpp_reduction_factor,  # Scale GeoJSON coords back to level 0 (full resolution)
        )
        gdf_contours.to_file(gdf_saveto, driver="GeoJSON")
        self.gdf_contours = gdf_contours  # Update WSI object's contours
        self.tissue_seg_path = gdf_saveto  # Update WSI object's tissue_seg_path

        # Draw the contours on the thumbnail image
        contours_saveto = os.path.join(job_dir, "contours", f"{self.name}.jpg")
        # Use the already loaded thumbnail_np (which is RGB)
        # Scale for overlay_gdf_on_thumbnail is from GeoJSON Level0 coords to thumbnail pixels.
        # So scale = thumbnail_width / self.width (or thumbnail_height / self.height)
        overlay_gdf_on_thumbnail(
            gdf_contours, thumbnail_np, contours_saveto, thumbnail_width / self.width
        )

        return gdf_saveto

    # These methods are abstract in the base class and implemented by subclasses (OpenSlideWSI, ImageWSI, CuCIMWSI)
    @abstractmethod
    def _fetch_mpp(self, custom_mpp_keys: Optional[List[str]] = None) -> float:
        """Abstract method to fetch MPP from backend-specific properties."""
        pass

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        """Abstract method to get dimensions from backend."""
        pass

    @abstractmethod
    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        """Abstract method to get a thumbnail from backend."""
        pass

    @abstractmethod
    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = "pil",
    ) -> Union[Image.Image, np.ndarray]:
        """Abstract method to read a region from backend."""
        pass

    @abstractmethod
    def level_dimensions(self) -> List[Tuple[int, int]]:
        """Abstract method to get all level dimensions from backend."""
        pass

    @abstractmethod
    def level_downsamples(self) -> List[float]:
        """Abstract method to get all level downsamples from backend."""
        pass

    @abstractmethod
    def get_best_level_and_custom_downsample(
        self, downsample: float, tolerance: float = 0.01
    ) -> Tuple[int, float]:
        """Abstract method to determine the best level and custom downsample."""
        pass

    @abstractmethod
    def close(self):
        """Abstract method to close the WSI object and release resources."""
        pass

    def extract_tissue_coords(
        self,
        target_mag: int,
        patch_size: int,
        save_coords: str,
        overlap: int = 0,
        min_tissue_proportion: float = 0.0,
    ) -> str:
        """
        The `extract_tissue_coords` function extracts patch coordinates
        from tissue regions in the WSI. It generates coordinates of patches at the specified
        magnification and saves the results in an HDF5 file.

        Args:
        -----
        target_mag : int
            Target magnification level for the patches.
        patch_size : int
            Size of each patch at the target magnification.
        save_coords : str
            Directory path to save the extracted coordinates.
        overlap : int, optional
            Overlap between patches in pixels. Defaults to 0.
        min_tissue_proportion: float, optional
            Minimum proportion of the patch under tissue to be kept. Defaults to 0.

        Returns:
        --------
        str:
            The absolute file path to the saved HDF5 file containing the patch coordinates.
        """

        self._lazy_initialize()  # Ensure WSI object is fully loaded

        # Ensure self.mpp and self.mag are available
        if self.mpp is None or self.mag is None:
            raise ValueError(
                f"WSI {self.name} does not have MPP or magnification information. Cannot extract tissue coordinates."
            )

        patcher = self.create_patcher(
            patch_size=patch_size,
            src_mag=self.mag,  # Use WSI's inferred or provided base magnification
            dst_mag=target_mag,
            mask=self.gdf_contours,  # Use the loaded or newly segmented contours
            coords_only=True,
            overlap=overlap,
            min_tissue_proportion=min_tissue_proportion,
        )

        coords_to_keep = np.array([(x, y) for x, y in patcher])

        if len(coords_to_keep) == 0:
            warnings.warn(
                f"No patches found for {self.name} after filtering. Coordinates file will be empty."
            )
            # For empty patches, return an empty array with appropriate shape
            coords_to_keep = np.empty((0, 2), dtype=np.int32)

        # Prepare assets for saving
        # patch_size_level0 is the size of the patch in pixels at Level 0 of the WSI
        patch_size_level0 = int(patch_size * self.mag / target_mag)

        assets = {"coords": coords_to_keep}
        attributes = {
            "patch_size": patch_size,  # Size of patch at target_mag
            "patch_size_level0": patch_size_level0,  # Size of patch at level 0
            "level0_magnification": self.mag,
            "target_magnification": target_mag,
            "overlap": overlap,
            "name": self.name,
            "savetodir": save_coords,  # This attribute is for context, not a path
        }

        # Save the assets and attributes to an hdf5 file
        out_fname = os.path.join(save_coords, "patches", str(self.name) + "_patches.h5")
        save_h5(out_fname, assets=assets, attributes={"coords": attributes}, mode="w")

        return out_fname

    def visualize_coords(self, coords_path: str, save_patch_viz: str) -> str:
        """
        The `visualize_coords` function overlays patch coordinates computed by the WSIPatcher
        onto a scaled thumbnail of the WSI. It creates a visualization of the extracted patches
        and saves it as an image file.

        Args:
        -----
        coords_path : str
            Path to the file containing the patch coordinates.
        save_patch_viz : str
            Directory path to save the visualization image.

        Returns:
        --------
        str:
            The file path to the saved visualization image.
        """

        self._lazy_initialize()  # Ensure WSI object is fully loaded

        try:
            coords_attrs, coords = read_coords(
                coords_path
            )  # Coords are ALWAYS wrt. level 0 of the slide.
            patch_size = coords_attrs.get("patch_size")
            level0_magnification = coords_attrs.get("level0_magnification")
            target_magnification = coords_attrs.get("target_magnification")

            if any(
                val is None
                for val in [patch_size, level0_magnification, target_magnification]
            ):
                raise KeyError(
                    "Missing essential attributes in coords_attrs (patch_size, level0_magnification, target_magnification)."
                )
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(
                f"Cannot read using new aegis coords format ({str(e)}). Trying with legacy CLAM/Fishing-Rod format."
            )
            # Fallback to legacy format
            try:
                patch_size, patch_level, custom_downsample, coords = read_coords_legacy(
                    coords_path
                )
                level0_magnification = self.mag
                if self.mag is None or not self.level_downsamples:
                    raise ValueError(
                        "WSI object's magnification or level_downsamples not initialized for legacy coords."
                    )
                target_magnification = int(
                    self.mag / (self.level_downsamples[patch_level] * custom_downsample)
                )
            except Exception as legacy_e:
                raise RuntimeError(
                    f"Failed to read coordinates from {coords_path} in both new and legacy formats: {legacy_e}"
                ) from legacy_e

        patcher = self.create_patcher(
            patch_size=patch_size,  # This is the size at target_magnification
            src_mag=level0_magnification,  # This is the base magnification of the coordinates
            dst_mag=target_magnification,  # This is the magnification at which patch_size is defined
            custom_coords=coords,  # Use the loaded coordinates
            coords_only=True,  # For visualization, we just need coordinates, not image data
            mask=self.gdf_contours,  # Pass existing contours for drawing visualization
        )

        img_pil = patcher.visualize()  # This method in WSIPatcher returns PIL Image

        # Save visualization
        os.makedirs(save_patch_viz, exist_ok=True)
        viz_coords_path = os.path.join(save_patch_viz, f"{self.name}.jpg")
        img_pil.save(viz_coords_path)
        return viz_coords_path

    @torch.inference_mode()
    def extract_patch_features(
        self,
        patch_encoder: BasePatchEncoder,  # Use the new base class for type hinting
        coords_path: str,
        save_features: str,
        device: str = "cuda:0",
        saveas: str = "h5",
        batch_limit: int = 512,
    ) -> str:
        """
        The `extract_patch_features` function extracts feature embeddings
        from the WSI using a specified patch encoder. It processes the patches as specified
        in the coordinates file and saves the features in the desired format.

        Args:
        -----
        patch_encoder : BasePatchEncoder (torch.nn.Module)
            The model used for feature extraction. Must be an instance of `BasePatchEncoder`
            (from `aegis.feature_extraction.models.patch_encoders.factory`).
        coords_path : str
            Path to the file containing patch coordinates.
        save_features : str
            Directory path to save the extracted features.
        device : str, optional
            Device to run feature extraction on (e.g., 'cuda:0'). Defaults to 'cuda:0'.
        saveas : str, optional
            Format to save the features ('h5' or 'pt'). Defaults to 'h5'.
        batch_limit : int, optional
            Maximum batch size for feature extraction. Defaults to 512.

        Returns:
        --------
        str:
            The absolute file path to the saved feature file in the specified format.
        """

        self._lazy_initialize()  # Ensure WSI object is fully loaded

        # Ensure patch_encoder is on correct device and in eval mode
        patch_encoder.to(device)
        patch_encoder.eval()

        precision = getattr(
            patch_encoder, "precision", torch.float32
        )  # Get precision from encoder, default to float32
        patch_transforms = patch_encoder.eval_transforms

        # Read coordinates and their attributes
        try:
            coords_attrs, coords = read_coords(coords_path)
            patch_size = coords_attrs.get("patch_size")
            level0_magnification = coords_attrs.get("level0_magnification")
            target_magnification = coords_attrs.get("target_magnification")
            if any(
                val is None
                for val in [patch_size, level0_magnification, target_magnification]
            ):
                raise KeyError(
                    "Missing essential attributes in coords_attrs (patch_size, level0_magnification, target_magnification)."
                )
        except (KeyError, FileNotFoundError, ValueError) as e:
            warnings.warn(
                f"Cannot read using new aegis coords format ({str(e)}). Trying with legacy CLAM/Fishing-Rod format."
            )
            # Fallback to legacy format
            try:
                patch_size, patch_level, custom_downsample, coords = read_coords_legacy(
                    coords_path
                )
                level0_magnification = (
                    self.mag
                )  # Use WSI's native mag for level0_magnification
                if self.mag is None or not self.level_downsamples:
                    raise ValueError(
                        "WSI object's magnification or level_downsamples not initialized for legacy coords."
                    )
                target_magnification = int(
                    self.mag / (self.level_downsamples[patch_level] * custom_downsample)
                )
            except Exception as legacy_e:
                raise RuntimeError(
                    f"Failed to read coordinates from {coords_path} in both new and legacy formats: {legacy_e}"
                ) from legacy_e

        # Create patcher for feature extraction
        patcher = self.create_patcher(
            patch_size=patch_size,  # This is the size at target_magnification
            src_mag=level0_magnification,  # Base magnification of the coordinates
            dst_mag=target_magnification,  # Magnification at which patches are effectively extracted/resized
            custom_coords=coords,  # Use the loaded coordinates
            coords_only=False,  # We need the actual image data
            pil=True,  # Patches as PIL Image to be processed by torchvision transforms
            mask=self.gdf_contours,  # For consistent behavior and potential visualization (though not used for filtering here)
        )

        dataset = WSIPatcherDataset(patcher, patch_transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_limit,
            num_workers=get_num_workers(batch_limit, max_workers=self.max_workers),
            pin_memory=True,
        )

        features = []
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            with torch.autocast(
                device_type=device.split(":")[0],
                dtype=precision,
                enabled=(precision != torch.float32),
            ):
                batch_features = patch_encoder(imgs)
            features.append(batch_features.cpu().numpy())

        # Concatenate features
        if not features:
            # Handle case where no patches were processed (e.g., empty slide after filtering)
            warnings.warn(
                f"No features extracted for {self.name}. Returning empty feature array."
            )
            feature_dim = getattr(patch_encoder, "embedding_dim", None)
            if feature_dim is None:
                # Attempt to get feature_dim from a dummy forward pass if not in attribute
                # This can be risky if model expects specific input shapes
                try:
                    dummy_input = (
                        patch_transforms(Image.new("RGB", (patch_size, patch_size)))
                        .unsqueeze(0)
                        .to(device)
                    )
                    feature_dim = patch_encoder(dummy_input).shape[-1]
                    warnings.warn(
                        f"Inferred empty feature dim as {feature_dim} for {self.name}"
                    )
                except Exception:
                    feature_dim = 0  # Cannot infer, return 0
            features_np = np.empty((0, feature_dim), dtype=np.float32)
        else:
            features_np = np.concatenate(features, axis=0)

        # Save the features to disk
        os.makedirs(save_features, exist_ok=True)
        out_fp = os.path.join(save_features, f"{self.name}.{saveas}")

        if saveas == "h5":
            save_h5(
                out_fp,
                assets={
                    "features": features_np,
                    "coords": coords,  # Save original level 0 coordinates
                },
                attributes={
                    "features": {"name": self.name, "savetodir": save_features},
                    "coords": coords_attrs,  # Save coordinate attributes
                },
                mode="w",
            )
        elif saveas == "pt":
            torch.save(
                {
                    "features": torch.from_numpy(features_np),
                    "coords": torch.from_numpy(coords),
                    "coords_attrs": coords_attrs,
                },
                out_fp,
            )
        else:
            raise ValueError(
                f'Invalid save_features_as: {saveas}. Only "h5" and "pt" are supported.'
            )

        return out_fp

    @torch.inference_mode()
    def extract_slide_features(
        self,
        patch_features_path: str,  # Path to the H5/PT file containing patch features and coords
        slide_encoder: BaseSlideEncoder,  # Use the new base class for type hinting
        save_features: str,  # Directory to save the final slide-level features
        device: str = "cuda",
    ) -> str:
        """
        Extract slide-level features by encoding patch-level features using a pretrained slide encoder.

        This function processes patch-level features extracted from a whole-slide image (WSI) and
        generates a single feature vector representing the entire slide. The extracted features are
        saved to a specified directory in HDF5 format.

        Args:
            patch_features_path (str): Path to the H5/PT file containing patch-level features and coordinates.
            slide_encoder (BaseSlideEncoder): Pretrained slide encoder model for generating slide-level features.
            save_features : str
            Directory where the extracted slide features will be saved.
            device (str, optional): Device to run computations on (e.g., 'cuda', 'cpu'). Defaults to 'cuda'.

        Returns:
            str: The absolute path to the slide-level features.

        Workflow:
            1. Load the pretrained slide encoder model and set it to evaluation mode.
            2. Load patch-level features and corresponding coordinates from the provided HDF5 file.
            3. Convert loaded data into tensors and move to the specified device.
            4. Generate slide-level features using the slide encoder.
            5. Save the slide-level features and associated metadata (e.g., original patch coordinates) in an HDF5 file.
            6. Return the path to the saved slide features.

        Notes:
            - The `patch_features_path` must point to a valid H5 or PT file containing datasets named `features` and `coords`.
            - The saved HDF5 file includes both the slide-level features and metadata such as patch coordinates.
            - Automatic mixed precision is enabled if the slide encoder supports precision lower than `torch.float32`.
        """
        import h5py

        # Set the slide encoder model to device and eval
        slide_encoder.to(device)
        slide_encoder.eval()

        # Determine file type and load accordingly
        file_extension = os.path.splitext(patch_features_path)[1].lower()
        if file_extension == ".h5":
            with h5py.File(patch_features_path, "r") as f:
                coords = f["coords"][:]
                patch_features = f["features"][:]
                coords_attrs = dict(f["coords"].attrs)
        elif file_extension == ".pt":
            loaded_data = torch.load(patch_features_path, map_location="cpu")
            patch_features = loaded_data["features"].numpy()
            coords = loaded_data["coords"].numpy()
            coords_attrs = loaded_data.get(
                "coords_attrs", {}
            )  # Get attributes if saved
        else:
            raise ValueError(
                f"Unsupported patch feature file format: {file_extension}. Only .h5 and .pt are supported."
            )

        # Convert to tensors for model input. Add batch dimension (B=1 for single slide)
        patch_features_tensor = (
            torch.from_numpy(patch_features).float().unsqueeze(0).to(device)
        )
        coords_tensor = torch.from_numpy(coords).to(device)
        if (
            coords_tensor.dtype == torch.float64
        ):  # Ensure coords are not float64 if not needed
            coords_tensor = coords_tensor.float()
        coords_tensor = coords_tensor.unsqueeze(0)  # Add batch dimension

        # Prepare input batch dictionary
        batch = {
            "features": patch_features_tensor,
            "coords": coords_tensor,
            "attributes": coords_attrs,  # Pass coordinate attributes, as some slide encoders need them (e.g., GigaPath, Titan)
        }

        # Generate slide-level features with autocast
        with torch.autocast(
            device_type=device.split(":")[0],
            dtype=slide_encoder.precision,
            enabled=(slide_encoder.precision != torch.float32),
        ):
            features = slide_encoder(
                batch, device
            )  # Call the slide encoder's forward method

        # Squeeze batch dim and convert to numpy
        features_np = (
            features.float().cpu().numpy().squeeze()
        )  # Ensure float32 for numpy and remove batch dim if present

        # Save slide-level features
        os.makedirs(save_features, exist_ok=True)
        save_path = os.path.join(
            save_features, f"{self.name}.h5"
        )  # Always save slide features as h5

        save_h5(
            save_path,
            assets={
                "features": features_np,
                "coords": coords,  # Keep original patch coords for traceability
            },
            attributes={
                "features": {
                    "name": self.name,
                    "savetodir": save_features,
                    "encoder_name": slide_encoder.enc_name,
                    "embedding_dim": slide_encoder.embedding_dim,
                },
                "coords": coords_attrs,  # Save original coordinate attributes
            },
            mode="w",
        )

        return save_path

    def release(self) -> None:
        """
        Release internal data (CPU/GPU/memory) and clear heavy references in the WSI instance.
        Call this method after you're done processing to avoid memory/GPU leaks.
        """
        # Clear backend image object

        if hasattr(self, "close"):
            self.close()  # Calls the backend-specific close method

        # Clear generic image object reference if exists
        if hasattr(self, "img"):
            self.img = None

        # Clear segmentation results and coordinates
        for attr in ["gdf_contours", "tissue_seg_path"]:
            if hasattr(self, attr):
                setattr(self, attr, None)

        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
