from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import openslide
from PIL import Image

from aegis.feature_extraction.wsi.base import WSI, ReadMode


class OpenSlideWSI(WSI):
    def __init__(
        self, slide_path: str, image_source: openslide.OpenSlide = None, **kwargs
    ) -> None:
        """
        Initialize an OpenSlideWSI instance.

        Parameters
        ----------
        slide_path : str
            Path to the WSI file.
        image_source : openslide.OpenSlide, optional
            An already opened OpenSlide object. If None, a new object will be created from slide_path.
        **kwargs : dict
            Keyword arguments forwarded to the base `WSI` class.
        """
        self.img = image_source
        super().__init__(slide_path=slide_path, **kwargs)

    def _lazy_initialize(self) -> None:
        """
        Lazily initialize the WSI using OpenSlide.

        This method opens a whole-slide image using the OpenSlide backend, extracting
        key metadata including dimensions, magnification, and multiresolution pyramid
        information. If a tissue segmentation mask is provided, it is also loaded.

        Raises
        ------
        FileNotFoundError
            If the WSI file or the tissue segmentation mask cannot be found.
        Exception
            If an unexpected error occurs during WSI initialization.

        Notes
        -----
        After initialization, the following attributes are set:
        - `width` and `height`: spatial dimensions of the base level.
        - `dimensions`: (width, height) tuple from the highest resolution.
        - `level_count`: number of resolution levels in the image pyramid.
        - `level_downsamples`: downsampling factors for each level.
        - `level_dimensions`: image dimensions at each level.
        - `properties`: metadata dictionary from OpenSlide.
        - `mpp`: microns per pixel, inferred if not manually specified.
        - `mag`: estimated magnification level.
        - `gdf_contours`: loaded from `tissue_seg_path` if provided.
        """
        if self._is_initialized:
            return

        try:
            if self.img is None:
                self.img = openslide.OpenSlide(self.slide_path)

            # Set OpenSlide attributes
            self.dimensions = self.img.dimensions
            self.width, self.height = self.dimensions
            self.level_count = self.img.level_count
            self.level_downsamples = self.img.level_downsamples
            self.level_dimensions = self.img.level_dimensions
            self.properties = self.img.properties

            # Fetch MPP and Magnification only if not already provided in constructor
            if self.mpp is None:
                self.mpp = self._fetch_mpp(self.custom_mpp_keys)
            if self.mag is None:
                self.mag = self._fetch_magnification(self.custom_mpp_keys)

            # Replicate contour loading logic from base class
            if (
                self.tissue_seg_path is not None
                and os.path.exists(self.tissue_seg_path)
                and self.gdf_contours is None
            ):
                try:
                    self.gdf_contours = gpd.read_file(self.tissue_seg_path)
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Failed to load GeoJSON from {self.tissue_seg_path}: {e}. Segmenting will proceed without pre-loaded contours."
                    )
                    self.gdf_contours = None
                    self.tissue_seg_path = None
            elif self.tissue_seg_path is None or not os.path.exists(
                self.tissue_seg_path
            ):
                self.gdf_contours = None

            self._is_initialized = True  # Mark as fully initialized at the end

        except openslide.OpenSlideError as ose:
            raise RuntimeError(
                f"Failed to open WSI with OpenSlide: {self.slide_path}. Error: {ose}"
            ) from ose
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize WSI with OpenSlide: {self.slide_path}. Error: {e}"
            ) from e

    def _fetch_mpp(self, custom_mpp_keys: Optional[List[str]] = None) -> float:
        """
        Retrieve microns per pixel (MPP) from OpenSlide metadata.

        Parameters
        ----------
        custom_mpp_keys : list of str, optional
            Additional metadata keys to check for MPP.

        Returns
        -------
        float
            MPP value in microns per pixel.

        Raises
        ------
        ValueError
            If MPP cannot be determined from metadata.
        """
        mpp_keys = [
            openslide.PROPERTY_NAME_MPP_X,  # Standard OpenSlide MPP key for X (also for Y)
            "openslide.mirax.MPP",  # Mirax specific
            "aperio.MPP",  # Aperio specific
            "hamamatsu.XResolution",  # Hamamatsu specific, needs conversion
            "openslide.comment",  # Sometimes embedded in comments
        ]

        if custom_mpp_keys:
            mpp_keys.extend(custom_mpp_keys)

        for key in mpp_keys:
            if key in self.properties:
                try:
                    mpp_x = float(self.properties[key])
                    if mpp_x > 0:  # Ensure MPP is positive
                        return round(mpp_x, 4)
                except ValueError:
                    continue  # Try next key

        # Fallback for TIFF resolution properties if MPP keys don't work
        x_resolution = self.properties.get("tiff.XResolution")
        unit = self.properties.get("tiff.ResolutionUnit")

        if x_resolution and unit:
            try:
                x_res_val = float(x_resolution)
                if x_res_val > 0:
                    if unit.lower() == "centimeter":
                        return round(10000 / x_res_val, 4)  # 10000 um/cm
                    elif unit.lower() == "inch":
                        return round(25400 / x_res_val, 4)  # 25400 um/inch
            except ValueError:
                pass

        raise ValueError(
            f"Unable to extract MPP from slide metadata for: '{self.slide_path}'.\n"
            "Suggestions:\n"
            "- Provide `custom_mpp_keys` to specify metadata keys to look for.\n"
            "- Set the MPP explicitly via the class constructor (`mpp` argument).\n"
        )

    def _fetch_magnification(self, custom_mpp_keys: Optional[List[str]] = None) -> int:
        """
        Retrieve estimated magnification from metadata or calculated from MPP.

        Parameters
        ----------
        custom_mpp_keys : list of str, optional
            Keys to aid in computing magnification from MPP (passed to _fetch_mpp).

        Returns
        -------
        int
            Estimated magnification.

        Raises
        -------
        ValueError
            If magnification cannot be determined.
        """
        # First try to get from OpenSlide's OBJECTIVE_POWER property
        metadata_mag = self.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if metadata_mag is not None:
            try:
                mag_val = int(
                    float(metadata_mag)
                )  # Cast to float first, then int, for robustness
                if mag_val > 0:
                    return mag_val
            except ValueError:
                pass  # If it's not a valid number, continue to next method

        # Then try to infer from MPP using base WSI's helper, if MPP is available
        inferred_mag_from_mpp = super()._fetch_magnification(custom_mpp_keys)
        if inferred_mag_from_mpp is not None:
            return inferred_mag_from_mpp

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
        Extract a specific region from the whole-slide image (WSI).

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
            Extracted image region in the specified format.

        Raises
        ------
        ValueError
            If `read_as` is not one of 'pil' or 'numpy'.
        """
        self._lazy_initialize()  # Ensure img is loaded

        # openslide.OpenSlide.read_region returns PIL.Image
        region_pil = self.img.read_region(location, level, size).convert("RGB")

        if read_as == "pil":
            return region_pil
        elif read_as == "numpy":
            return np.array(region_pil)
        else:
            raise ValueError(
                f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'."
            )

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Return the dimensions (width, height) of the WSI at level 0.

        Returns
        -------
        tuple of int
            (width, height) in pixels.
        """
        self._lazy_initialize()  # Ensure img is loaded
        return self.dimensions

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        """
        Generate a thumbnail of the WSI.

        Parameters
        ----------
        size : tuple of int
            Desired (width, height) of the thumbnail.

        Returns
        -------
        PIL.Image.Image
            RGB thumbnail as a PIL Image.
        """
        self._lazy_initialize()  # Ensure img is loaded
        return self.img.get_thumbnail(size).convert("RGB")

    def level_dimensions(self) -> List[Tuple[int, int]]:
        """
        Gets the dimensions of each level in the WSI.

        Returns
        -------
        List[Tuple[int, int]]: A list of (width, height) tuples for each level.
        """
        self._lazy_initialize()  # Ensure img is loaded
        return self.img.level_dimensions

    def level_downsamples(self) -> List[float]:
        """
        Gets the downsample factor for each level in the WSI.

        Returns
        -------
        List[float]: A list of downsample factors for each level relative to level 0.
        """
        self._lazy_initialize()  # Ensure img is loaded
        return self.img.level_downsamples

    def get_best_level_and_custom_downsample(
        self, downsample: float, tolerance: float = 0.01
    ) -> Tuple[int, float]:
        """
        Determines the best OpenSlide level and custom downsample factor to approximate a desired downsample value.

        Args:
            downsample (float): The desired total downsample factor relative to level 0.
            tolerance (float, optional): Tolerance for matching existing downsample levels. Defaults to 0.01.

        Returns:
            Tuple[int, float]: A tuple containing the best level index and the additional
                               custom downsample factor needed at that level.
        Raises:
            ValueError: If no suitable level can be found (should not happen for valid OpenSlide).
        """
        self._lazy_initialize()  # Ensure img is loaded

        # OpenSlide's get_best_level_for_downsample returns the level with the highest resolution
        # that is greater than or equal to the desired downsample factor.
        best_level_index = self.img.get_best_level_for_downsample(downsample)

        # The actual downsample at that level
        actual_level_downsample = self.img.level_downsamples[best_level_index]

        # Calculate the additional custom_downsample factor needed
        # If actual_level_downsample is exactly `downsample`, custom_downsample will be 1.0.
        # If `downsample` is higher (more zoomed in) than `actual_level_downsample`,
        # then we need to upsample, so `custom_downsample` will be > 1.0.
        # If `downsample` is lower (more zoomed out) than `actual_level_downsample`,
        # then we need to downsample further, so `custom_downsample` will be < 1.0.

        if actual_level_downsample > 0:  # Avoid division by zero
            custom_downsample = downsample / actual_level_downsample
        else:
            raise ValueError(
                f"Invalid level downsample factor {actual_level_downsample} at level {best_level_index} for slide {self.name}."
            )

        return best_level_index, custom_downsample

    def close(self):
        """
        Close the OpenSlide object and release its resources.
        """
        if self.img is not None:
            self.img.close()
            self.img = None
            self._is_initialized = False  # Mark as uninitialized
