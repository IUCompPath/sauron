from __future__ import annotations

import warnings
from typing import Any, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
from PIL import Image

from aegis.feature_extraction.utils.io import CuImageWarning  # For warning
from aegis.feature_extraction.wsi.base import WSI, ReadMode
from aegis.feature_extraction.wsi.patching import CuImageWSIPatcher  # For type hinting


class CuCIMWSI(WSI):
    def __init__(
        self, image_source: Any, **kwargs
    ) -> None:  # Use Any for cucim.CuImage due to import issues
        # Check if cucim is actually available before proceeding, to provide clearer errors
        try:
            from cucim import CuImage
        except ImportError:
            # Re-raise if cucim is chosen but not installed
            raise ImportError(
                "cuCIM is not installed. Cannot use CuCIMWSI backend. "
                "Install with `pip install cucim cupy-cuda12x` (adjust CUDA version)."
            )

        if not isinstance(image_source, CuImage):
            # If image_source is a path, try to open it with CuImage
            if isinstance(image_source, str):
                try:
                    self.img = CuImage(image_source)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to open WSI with CuImage from path '{image_source}': {e}"
                    )
            else:
                raise ValueError(
                    f"CuCIMWSI expects a cucim.CuImage object or a string path, but got {type(image_source)}."
                )
        else:
            self.img = image_source  # Store the CuImage object directly

        super().__init__(
            **kwargs
        )  # Call base WSI init, will trigger _lazy_initialize if lazy_init=False

    def _lazy_initialize(self) -> None:
        """
        Lazily load the whole-slide image (WSI) and its metadata using CuCIM.

        This method performs deferred initialization by reading the WSI file
        only when needed. It also retrieves key metadata such as dimensions,
        magnification, and microns-per-pixel (MPP). If a tissue segmentation
        mask is available, it is also loaded.

        Raises
        ------
        FileNotFoundError
            If the WSI file or required segmentation mask is missing.
        Exception
            For any other errors that occur while initializing the WSI.

        Notes
        -----
        After initialization, the following attributes are set:
        - `width` and `height`: spatial dimensions of the WSI.
        - `mpp`: microns per pixel, inferred if not already set.
        - `mag`: estimated magnification level of the image.
        - `level_count`, `level_downsamples`, and `level_dimensions`: multiresolution pyramid metadata.
        - `properties`: raw metadata from the image.
        - `gdf_contours`: tissue mask contours, if applicable.
        """

        if (
            not self._is_initialized
        ):  # Only proceed if base init didn't already mark as initialized
            try:
                # self.img should already be set from __init__ for CuCIMWSI
                # If not provided, it means this WSI object was created via `load_wsi(path)`
                # which would provide the CuImage object to __init__.
                if self.img is None:
                    from cucim import CuImage

                    self.img = CuImage(self.slide_path)

                # CuImage's size() returns (height, width), unlike OpenSlide's dimensions (width, height)
                self.dimensions = (
                    self.img.size()[1],
                    self.img.size()[0],
                )  # width, height
                self.width, self.height = self.dimensions

                # CuImage's resolutions dictionary
                self.level_count = self.img.resolutions["level_count"]
                self.level_downsamples = self.img.resolutions["level_downsamples"]
                self.level_dimensions = self.img.resolutions["level_dimensions"]
                self.properties = self.img.metadata

                # Fetch MPP and Magnification only if not already provided in constructor
                if self.mpp is None:
                    self.mpp = self._fetch_mpp(self.custom_mpp_keys)
                if self.mag is None:
                    self.mag = self._fetch_magnification(self.custom_mpp_keys)

                self._is_initialized = True  # Mark as fully initialized

            except Exception as e:
                # Issue CuImageWarning if cucim is technically installed but fails to load a specific file
                warnings.warn(
                    f"Failed to initialize WSI using CuCIM for '{self.slide_path}'. Falling back may not be possible. Error: {e}",
                    CuImageWarning,
                )
                raise RuntimeError(
                    f"Failed to initialize WSI using CuCIM: {self.slide_path}. Error: {e}"
                ) from e

    def _fetch_mpp(self, custom_keys: Optional[List[str]] = None) -> float:
        """
        Fetch the microns per pixel (MPP) from CuImage metadata.

        Parameters
        ----------
        custom_keys : list of str, optional
            Optional list of custom metadata keys (e.g., 'mpp_x', 'mpp_y') to check first.

        Returns
        -------
        float
            MPP value in microns per pixel.

        Raises
        -------
        ValueError
            If MPP cannot be determined from metadata.
        """
        import json
        import warnings

        def try_parse_float(val: Any) -> Optional[float]:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        # CuCIM metadata can be a JSON string or a dictionary
        metadata = self.img.metadata
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                warnings.warn(
                    f"CuCIM metadata for {self.name} is a malformed JSON string. Will proceed with direct key access."
                )
                metadata = {}  # Treat as empty dict if malformed

        # Flatten nested CuCIM metadata for easier access
        flat_meta = {}

        def flatten_dict(d: dict, parent_key: str = ""):
            for k, v in d.items():
                key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_dict(v, key)
                else:
                    flat_meta[key.lower()] = v

        # Only flatten if metadata is a dict; if it's a direct string, it's not nested dict.
        if isinstance(metadata, dict):
            flatten_dict(metadata)
        elif isinstance(
            metadata, str
        ):  # Handle cases where metadata might be a direct string value for MPP
            if try_parse_float(metadata) is not None:
                return try_parse_float(
                    metadata
                )  # If the whole metadata string IS the MPP

        mpp_x = None
        mpp_y = None

        # Prioritize custom keys if provided
        if custom_keys:
            # Assume custom_keys contains string keys like 'mpp_x', 'mpp_y' directly
            # It's better if custom_keys maps to the exact metadata path e.g. {'mpp_x': 'some.nested.mpp_x'}
            # For simplicity, if custom_keys provided, treat them as direct lookup keys in flat_meta
            for k in custom_keys:
                lower_k = k.lower()
                if mpp_x is None and lower_k in flat_meta:
                    mpp_x = try_parse_float(flat_meta[lower_k])
                if mpp_y is None and lower_k in flat_meta:
                    mpp_y = try_parse_float(flat_meta[lower_k])
                if mpp_x is not None and mpp_y is not None:
                    break  # Found both

        # Standard fallback keys used in SVS, NDPI, MRXS, etc. (often present in CuCIM metadata)
        # Note: CuCIM sometimes uses 'spacing' or 'resolution' directly for MPP
        fallback_keys = [
            "openslide.mpp-x",
            "openslide.mpp-y",
            "tiff.resolution-x",
            "tiff.resolution-y",
            "mpp",
            "spacing",
            "microns_per_pixel",
            "aperio.mpp",
            "hamamatsu.mpp",
            "metadata.resolutions.level[0].spacing",
            "metadata.resolutions.level[0].physical_size.0",  # Common for some OME-TIFF metadata
            "resolution",  # Simple key, sometimes used.
        ]

        for key in fallback_keys:
            lower_key = key.lower()
            if mpp_x is None and lower_key in flat_meta:
                mpp_x = try_parse_float(flat_meta[lower_key])
            if mpp_y is None and lower_key in flat_meta:  # Check for Y explicitly
                mpp_y = try_parse_float(flat_meta[lower_key])
            if mpp_x is not None and mpp_y is not None:
                break

        # Use same value for both axes if only one was found
        if mpp_x is not None and mpp_y is None:
            mpp_y = mpp_x
        if mpp_y is not None and mpp_x is None:
            mpp_x = mpp_y

        # Ensure positive and non-zero MPP
        if mpp_x is not None and mpp_y is not None and mpp_x > 0 and mpp_y > 0:
            return float((mpp_x + mpp_y) / 2)

        raise ValueError(
            f"Unable to extract MPP from CuCIM metadata for: '{self.slide_path}'.\n"
            "Suggestions:\n"
            "- Provide `custom_mpp_keys` with metadata key mappings for 'mpp_x' and 'mpp_y'.\n"
            "- Set the MPP manually when constructing the CuCIMWSI object (e.g., `mpp=0.25`)."
        )

    def _fetch_magnification(self, custom_mpp_keys: Optional[List[str]] = None) -> int:
        """
        Retrieve estimated magnification from CuCIM metadata or calculated from MPP.

        Parameters
        ----------
        custom_mpp_keys : list of str, optional
            Keys to aid in computing magnification from MPP.

        Returns
        -------
        int
            Estimated magnification.

        Raises
        -------
        ValueError
            If magnification cannot be determined.
        """
        # First try to infer from MPP using base WSI's helper, if MPP is available
        inferred_mag_from_mpp = super()._fetch_magnification(custom_mpp_keys)
        if inferred_mag_from_mpp is not None:
            return inferred_mag_from_mpp

        # Then try to get from CuCIM properties (less common to have a direct "magnification" property)
        # Assuming properties could be a dict
        if isinstance(self.properties, dict):
            metadata_mag = self.properties.get("objective_power")  # Common key
            if metadata_mag is not None:
                try:
                    mag_val = int(float(metadata_mag))  # Robust conversion
                    if mag_val > 0:
                        return mag_val
                except ValueError:
                    pass
            # Also check in flattened metadata if available
            flat_meta = {}
            if isinstance(self.properties, dict):

                def flatten_dict_recursive(d, parent_key=""):
                    for k, v in d.items():
                        new_key = f"{parent_key}.{k}" if parent_key else k
                        if isinstance(v, dict):
                            flatten_dict_recursive(v, new_key)
                        else:
                            flat_meta[new_key.lower()] = v

                flatten_dict_recursive(self.properties)
                if "objective_power" in flat_meta:
                    try:
                        mag_val = int(float(flat_meta["objective_power"]))
                        if mag_val > 0:
                            return mag_val
                    except ValueError:
                        pass

        raise ValueError(
            f"Unable to determine magnification from metadata for: {self.slide_path}"
        )

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = "pil",
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a specific region from the whole-slide image (WSI) using CuCIM.

        Parameters
        ----------
        location : Tuple[int, int]
            (x, y) coordinates of the top-left corner of the region to extract at level 0.
        level : int
            Pyramid level to read from.
        size : Tuple[int, int]
            (width, height) of the region to extract at the specified level.
        read_as : {'pil', 'numpy'}, optional
            Output format for the region:
            - 'pil': returns a PIL Image (default)
            - 'numpy': returns a NumPy array (H, W, 3)

        Returns
        -------
        Union[PIL.Image.Image, np.ndarray]
            The extracted region in the specified format.

        Raises
        -------
        ValueError
            If `read_as` is not one of the supported options.
        """
        self._lazy_initialize()  # Ensure img is loaded

        import cupy as cp

        # CuCIM's read_region returns a CuPy array. Convert to NumPy.
        # It's always (C, H, W) by default unless specified, but CuCIM also supports (H, W, C)
        # Default behavior: (C, H, W) is typical for `read_region` of images.
        # However, `read_region` for images usually outputs (H, W, C).
        # To be safe, specify `to_device='cpu'` which often returns NumPy.
        region_cp_or_np = self.img.read_region(
            location=location, level=level, size=size, to_device="cpu"
        )

        # Ensure it's a NumPy array and has the correct channel order (H, W, C)
        if isinstance(region_cp_or_np, cp.ndarray):
            region_np = cp.asnumpy(region_cp_or_np)
        else:  # Assumed to be numpy array already if to_device='cpu' worked
            region_np = region_cp_or_np

        # If image is (C, H, W), transpose to (H, W, C)
        if (
            region_np.ndim == 3
            and region_np.shape[0] < region_np.shape[1]
            and region_np.shape[0] < region_np.shape[2]
        ):
            region_np = np.transpose(region_np, (1, 2, 0))  # C H W -> H W C

        # Ensure 3 channels, handle RGBA by dropping alpha
        if region_np.shape[-1] == 4:
            region_np = region_np[:, :, :3]
        elif region_np.shape[-1] == 1:  # Convert grayscale to RGB
            region_np = np.stack([region_np.squeeze()] * 3, axis=-1)

        if read_as == "numpy":
            return region_np
        elif read_as == "pil":
            return Image.fromarray(region_np).convert("RGB")
        else:
            raise ValueError(
                f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'."
            )

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the (width, height) dimensions of the CuCIM-managed WSI at level 0.
        """
        self._lazy_initialize()  # Ensure img is loaded
        # CuImage's size() returns (height, width)
        return (self.img.size()[1], self.img.size()[0])

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail image of the WSI using CuCIM.

        Args:
        -----
        size : tuple[int, int]
            A tuple specifying the desired width and height of the thumbnail.

        Returns:
        --------
        Image.Image:
            The thumbnail as a PIL Image in RGB format.
        """
        self._lazy_initialize()  # Ensure img is loaded

        target_width, target_height = size

        # Compute desired downsample factor from original dimensions
        downsample_x = self.width / target_width
        downsample_y = self.height / target_height
        desired_downsample_factor_relative_to_level0 = max(downsample_x, downsample_y)

        # Get the best level index and the additional custom downsample needed from that level
        level, custom_downsample = self.get_best_level_and_custom_downsample(
            desired_downsample_factor_relative_to_level0
        )

        # Read the region at the chosen level. Size needs to be computed correctly for that level.
        level_actual_dims = self.level_dimensions()[level]
        # Read the full content of this level, then resize in CPU
        region_np = self.read_region(
            location=(0, 0),  # Read from top-left of the level
            level=level,
            size=level_actual_dims,  # Read the whole level
            read_as="numpy",  # Get as numpy to resize with OpenCV (or PIL)
        )

        # Resize to target size. CuCIM read_region sometimes has issues with direct target size.
        # Using PIL resize from numpy.
        thumbnail_pil = Image.fromarray(region_np).convert("RGB")
        thumbnail_pil = thumbnail_pil.resize(size, resample=Image.Resampling.LANCZOS)

        return thumbnail_pil

    def level_dimensions(self) -> List[Tuple[int, int]]:
        """
        Gets the dimensions of each level in the WSI from CuCIM.
        """
        self._lazy_initialize()
        return self.img.resolutions["level_dimensions"]

    def level_downsamples(self) -> List[float]:
        """
        Gets the downsample factor for each level in the WSI from CuCIM.
        """
        self._lazy_initialize()
        return self.img.resolutions["level_downsamples"]

    def get_best_level_and_custom_downsample(
        self,
        downsample: float,  # desired downsample factor relative to level 0 (e.g. 2.0 for 2x downsample from level 0)
        tolerance: float = 0.01,  # Not strictly used, but for consistency
    ) -> Tuple[int, float]:
        """
        Determines the best CuCIM level and custom downsample factor to approximate a desired downsample value.

        Args:
            downsample (float): The desired total downsample factor relative to level 0.
            tolerance (float, optional): Tolerance for matching existing downsample levels. Defaults to 0.01.

        Returns:
            Tuple[int, float]: A tuple containing the best level index and the additional
                               custom downsample factor needed at that level.
        """
        self._lazy_initialize()  # Ensure img is loaded

        level_downsamples = self.img.resolutions["level_downsamples"]

        # Find the smallest level_downsample factor that is >= desired `downsample`
        # This gives the level that is *just* high enough resolution.
        best_level_index = 0
        for i, level_ds in enumerate(level_downsamples):
            if level_ds >= downsample:
                best_level_index = i
                break

        actual_level_downsample = level_downsamples[best_level_index]

        # Calculate the custom downsample needed from this `best_level_index`
        # If actual_level_downsample is 4.0, and desired is 2.0, then custom_downsample = 2.0 / 4.0 = 0.5 (need to downsample by 0.5 at that level)
        # If actual_level_downsample is 2.0, and desired is 4.0, then custom_downsample = 4.0 / 2.0 = 2.0 (need to upsample by 2.0 at that level)
        if actual_level_downsample > 0:  # Avoid division by zero
            custom_downsample = downsample / actual_level_downsample
        else:
            raise ValueError(
                f"Invalid level downsample factor {actual_level_downsample} at level {best_level_index} for slide {self.name}."
            )

        return best_level_index, custom_downsample

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
        min_tissue_proportion: float = 0.0,
        pil: bool = False,
    ) -> CuImageWSIPatcher:
        """
        Creates a patcher for the cuCIM WSI.
        """
        # Call the base WSI's create_patcher. It will then call the appropriate WSIPatcher subclass.
        return CuImageWSIPatcher(
            wsi=self,  # Pass self (CuCIMWSI instance)
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

    def close(self):
        """
        Close the CuCIM object and release its resources.
        """
        if self.img is not None:
            self.img.close()
            self.img = None
            self._is_initialized = False  # Mark as uninitialized
