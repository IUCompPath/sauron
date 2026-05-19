from __future__ import annotations

from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
from PIL import Image

from aegis.feature_extraction.wsi.base import WSI, ReadMode
from aegis.feature_extraction.wsi.patching import NumpyWSIPatcher  # For type hinting


class ImageWSI(WSI):
    def __init__(self, image_source: Image.Image, **kwargs) -> None:
        """
        Initialize a WSI object from a standard PIL Image file (e.g., PNG, JPEG, etc.).

        Parameters
        ----------
        image_source : PIL.Image.Image
            An already opened PIL Image object.
        mpp : float
            Microns per pixel. Required since standard image formats do not store this metadata.
        **kwargs: other arguments to pass to base WSI constructor.
            - name : str, optional (inherited)
            - lazy_init : bool, default=True (inherited)

        Raises
        ------
        ValueError
            If the required 'mpp' argument is not provided.
        """
        mpp = kwargs.get("mpp")
        if mpp is None:
            raise ValueError(
                "Missing required argument `mpp`. Standard image formats (PNG, JPEG etc.) do not contain microns-per-pixel "
                "information, so you must specify it manually via the `ImageWSI` constructor's `mpp` argument (or in CSV for Processor)."
            )

        # Enable loading large images if MAX_IMAGE_PIXELS is set for PIL
        # This is usually set globally, but ensuring it here.
        Image.MAX_IMAGE_PIXELS = (
            None  # Disables decompression bomb limit if not set globally
        )

        self.img = image_source  # Store the PIL Image object directly
        super().__init__(
            **kwargs
        )  # Call base WSI init, will trigger _lazy_initialize if lazy_init=False

    def _lazy_initialize(self) -> None:
        """
        Lazily initialize the WSI using a standard PIL Image object.

        This method loads the image using PIL and extracts relevant metadata such as
        dimensions and magnification. It assumes a single-resolution image (no pyramid).
        If a tissue segmentation mask is available, it is also loaded.

        Raises
        ------
        FileNotFoundError
            If the WSI file (if re-opened) or tissue segmentation mask is not found.
        Exception
            If an unexpected error occurs during initialization.

        Notes
        -----
        After initialization, the following attributes are set:
        - `width` and `height`: dimensions of the image.
        - `dimensions`: (width, height) tuple of the image.
        - `level_downsamples`: set to `[1]` (single resolution).
        - `level_dimensions`: set to a list containing the image dimensions.
        - `level_count`: set to `1`.
        - `mag`: estimated magnification level.
        - `gdf_contours`: loaded from `tissue_seg_path`, if available.
        """

        super()._lazy_initialize()  # Call base class lazy init

        if (
            not self._is_initialized
        ):  # Only proceed if base init didn't already mark as initialized
            try:
                # self.img should already be set from __init__ for ImageWSI
                # If not provided, it means this WSI object was created via `load_wsi(path)`
                # which would provide the PIL Image object to __init__.
                if self.img is None:
                    # If somehow self.img is None, try to re-open the file.
                    # This might happen if `ImageWSI` was initialized lazily and then later closed.
                    self.img = Image.open(self.slide_path).convert("RGB")

                self.level_downsamples = [1.0]  # Single level image
                self.level_dimensions = [
                    (self.img.width, self.img.height)
                ]  # Store actual dimensions
                self.level_count = 1

                self.dimensions = self.get_dimensions()  # Get actual dimensions
                self.width, self.height = self.dimensions[0], self.dimensions[1]

                # Fetch MPP and Magnification only if not already provided in constructor
                # For ImageWSI, MPP *must* be provided in constructor, it cannot be fetched
                if self.mpp is None:
                    raise ValueError(
                        f"MPP for {self.name} is not available. ImageWSI requires MPP to be passed during initialization."
                    )
                # Magnification is inferred from the provided MPP
                if self.mag is None:
                    self.mag = self._fetch_magnification(self.custom_mpp_keys)

                self._is_initialized = True  # Mark as fully initialized
            except Exception as e:
                raise RuntimeError(
                    f"Error initializing WSI with PIL.Image backend: {self.slide_path}. Error: {e}"
                ) from e

    def _fetch_mpp(self, custom_mpp_keys: Optional[List[str]] = None) -> float:
        """
        For ImageWSI, MPP *must* be provided in the constructor. This method acts as a check.
        """
        if self.mpp is not None:
            return self.mpp
        else:
            # This should have been caught in __init__, but as a safeguard.
            raise ValueError(
                f"MPP not available for ImageWSI '{self.name}'. It must be provided during initialization."
            )

    def _fetch_magnification(self, custom_mpp_keys: Optional[List[str]] = None) -> int:
        """
        Retrieve estimated magnification from `self.mpp` (which must be set).
        """
        self._lazy_initialize()  # Ensure MPP is available
        inferred_mag_from_mpp = super()._fetch_magnification(custom_mpp_keys)
        if inferred_mag_from_mpp is not None:
            return inferred_mag_from_mpp
        else:
            raise ValueError(
                f"Unable to determine magnification for ImageWSI '{self.name}'. MPP is available but could not infer common magnification."
            )

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the dimensions (width, height) of the PIL Image.
        """
        self._lazy_initialize()  # Ensure img is loaded
        return self.img.size

    def get_thumbnail(self, size: Tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail of the PIL Image.

        Parameters
        ----------
        size : tuple of int
            Desired thumbnail size (width, height).

        Returns
        -------
        PIL.Image.Image
            RGB thumbnail image.
        """
        self._lazy_initialize()  # Ensure img is loaded
        img_copy = (
            self.img.copy()
        )  # Work on a copy to avoid modifying original in place
        img_copy.thumbnail(
            size, Image.Resampling.LANCZOS
        )  # Use LANCZOS for high-quality downsampling
        return img_copy.convert("RGB")

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = "pil",
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a specific region from a single-resolution image (e.g., JPEG, PNG, TIFF).

        Parameters
        ----------
        location : Tuple[int, int]
            (x, y) coordinates of the top-left corner of the region to extract (at level 0).
        level : int
            Pyramid level to read from. Only level 0 is supported for non-pyramidal images.
        size : Tuple[int, int]
            (width, height) of the region to extract.
        read_as : {'pil', 'numpy'}, optional
            Output format for the region:
            - 'pil': returns a PIL Image (default)
            - 'numpy': returns a NumPy array (H, W, 3)

        Returns
        -------
        Union[PIL.Image.Image, np.ndarray]
            Extracted image region in the specified format.

        Raises
        -------
        ValueError
            If `level` is not 0 or if `read_as` is not one of the supported options.
        """
        if level != 0:
            raise ValueError(
                "ImageWSI only supports reading at level=0 (no pyramid levels)."
            )

        self._lazy_initialize()  # Ensure img is loaded

        # Ensure the region is within bounds to prevent PIL errors for out-of-bounds crops
        x, y = location
        w, h = size

        x_end = min(x + w, self.width)
        y_end = min(y + h, self.height)

        # Adjust width and height to actual cropped size if bounds were hit
        actual_w = x_end - x
        actual_h = y_end - y

        if actual_w <= 0 or actual_h <= 0:
            # If the crop region is invalid (e.g., completely outside), return a black image
            if read_as == "pil":
                return Image.new("RGB", size, (0, 0, 0))
            else:
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        region = self.img.crop((x, y, x_end, y_end)).convert("RGB")

        # If the extracted region is smaller than requested (due to image boundary), pad it.
        if region.width != w or region.height != h:
            padded_region = Image.new("RGB", (w, h), (0, 0, 0))  # Black padding
            padded_region.paste(region, (0, 0))
            region = padded_region

        if read_as == "pil":
            return region
        elif read_as == "numpy":
            return np.array(region)
        else:
            raise ValueError(
                f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'."
            )

    def level_dimensions(self) -> List[Tuple[int, int]]:
        """
        Gets the dimensions of each level in the WSI. For ImageWSI, it's always just level 0.
        """
        self._lazy_initialize()
        return self.level_dimensions

    def level_downsamples(self) -> List[float]:
        """
        Gets the downsample factor for each level in the WSI. For ImageWSI, it's always just level 0 with 1.0.
        """
        self._lazy_initialize()
        return self.level_downsamples

    def get_best_level_and_custom_downsample(
        self,
        downsample: float,
        tolerance: float = 0.01,  # Not strictly used, but kept for signature consistency
    ) -> Tuple[int, float]:
        """
        Determines the best level and custom downsample for an ImageWSI.
        Since ImageWSI has only one level (level 0), it always returns level 0.
        The custom_downsample is the desired `downsample` factor itself.
        """
        self._lazy_initialize()
        return (
            0,
            downsample,
        )  # Always level 0, custom downsample is the requested factor

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
    ) -> NumpyWSIPatcher:  # ImageWSI internally treats images as numpy-like
        """
        Creates a patcher for the PIL Image WSI.
        """
        # Call the base WSI's create_patcher. It will then call the appropriate WSIPatcher subclass.
        # Since ImageWSI doesn't have OpenSlide or CuCIM specific features, it routes to NumpyWSIPatcher.
        return NumpyWSIPatcher(
            wsi=self,  # Pass self (ImageWSI instance)
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
        Close the internal PIL Image object to free memory.
        """
        if self.img is not None:
            self.img.close()
            self.img = None
            self._is_initialized = False  # Mark as uninitialized
