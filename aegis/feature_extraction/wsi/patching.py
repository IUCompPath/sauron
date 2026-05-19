from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
from PIL import Image
from shapely import Polygon
from shapely.validation import (
    make_valid as shapely_make_valid,  # Corrected: Import shapely_make_valid
)

# Prevent circular import if Base is not fully defined yet, for type hinting
if TYPE_CHECKING:
    from aegis.feature_extraction.wsi.base import WSI


class WSIPatcher:
    """Iterator class to handle patching, patch scaling and tissue mask intersection"""

    def __init__(
        self,
        wsi: "WSI",
        patch_size: int,  # Desired output patch size in pixels (e.g., 256 for 256x256)
        src_pixel_size: float,  # MPP of the WSI at level 0 (e.g., 0.25)
        dst_pixel_size: Optional[
            float
        ] = None,  # Desired MPP for output patches (e.g., 0.5 for 20x)
        src_mag: Optional[int] = None,  # Base magnification of the WSI (e.g., 40)
        dst_mag: Optional[int] = None,  # Desired output magnification (e.g., 20)
        overlap: int = 0,  # Overlap in pixels at the output patch size (dst_pixel_size/dst_mag)
        mask: Optional[gpd.GeoDataFrame] = None,
        coords_only: bool = False,
        custom_coords: Optional[
            np.ndarray
        ] = None,  # (N, 2) array of (x_level0, y_level0)
        min_tissue_proportion: float = 0.0,  # Min proportion of patch that must be tissue
        pil: bool = False,  # Return PIL Image or NumPy array
    ):
        """Initialize patcher, compute number of (masked) rows, columns.

        Args:
            wsi (WSI): WSI object to patch.
            patch_size (int): Patch width/height in pixels for the *output* patches (e.g., 256).
            src_pixel_size (float): MPP of the WSI at level 0.
            dst_pixel_size (float, optional): Desired MPP of the output patches. If None, `dst_mag` must be provided.
            src_mag (int, optional): Magnification of the WSI at level 0. If None, `self.wsi.mag` is used.
            dst_mag (int, optional): Target magnification for the output patches. If None, `dst_pixel_size` must be provided.
            overlap (int, optional): Overlap between patches in pixels (at `patch_size` resolution). Defaults to 0.
            mask (gpd.GeoDataFrame, optional): GeoDataFrame of Polygons for tissue regions. Patches are filtered by this mask. Defaults to None (no filtering).
            coords_only (bool, optional): If True, iterator yields only (x, y) coordinates. Otherwise, yields (image, x, y). Default to False.
            custom_coords (np.ndarray, optional): Pre-defined (N, 2) array of (x, y) coordinates at Level 0 to extract. If provided, grid generation and mask filtering are bypassed. Defaults to None.
            min_tissue_proportion (float, optional): Minimum proportion of the *output* patch (at `patch_size`) that must be tissue to be kept. Only applies if `mask` is provided. Between 0. and 1.0. Defaults to 0. (any tissue).
            pil (bool, optional): If True, `get_patch_at` returns PIL.Image. Otherwise, NumPy array. Defaults to False.
        """
        self.wsi = wsi
        self.overlap = overlap
        self.width, self.height = self.wsi.get_dimensions()  # WSI level 0 dimensions
        self.patch_size_output = (
            patch_size  # The final size of the patch returned by iterator
        )

        self.mask = mask
        self.current_index = 0
        self.coords_only = coords_only
        self.pil = pil
        self.min_tissue_proportion = min_tissue_proportion

        # --- Determine scaling from src to output resolution (pixel size or magnification) ---
        # Ensure either dst_pixel_size or dst_mag is provided if scaling is implied
        if dst_pixel_size is None and dst_mag is None:
            warnings.warn(
                "Neither dst_pixel_size nor dst_mag provided. Assuming no scaling is desired (dst_pixel_size = src_pixel_size, dst_mag = src_mag)."
            )
            self.dst_pixel_size = src_pixel_size
            self.dst_mag = src_mag
        elif dst_pixel_size is not None and dst_mag is not None:
            # If both are provided, check for consistency. Prefer pixel size for more exact control.
            if (
                abs(dst_pixel_size - (10 / dst_mag)) > 0.01
            ):  # Check for a small tolerance
                warnings.warn(
                    f"Inconsistent dst_pixel_size ({dst_pixel_size}) and dst_mag ({dst_mag}). Using dst_pixel_size."
                )
            self.dst_pixel_size = dst_pixel_size
            self.dst_mag = dst_mag
        elif dst_pixel_size is not None:
            self.dst_pixel_size = dst_pixel_size
            self.dst_mag = 10 / dst_pixel_size if dst_pixel_size > 0 else None
        elif dst_mag is not None:
            self.dst_mag = dst_mag
            self.dst_pixel_size = 10 / dst_mag if dst_mag > 0 else None

        if (
            self.dst_pixel_size is None
        ):  # Should not happen with above logic, but for safety
            raise ValueError("Could not determine destination pixel size.")

        # Calculate the overall downsample factor from WSI's level 0 MPP to the desired output MPP
        # Example: src_pixel_size=0.25 (40x), dst_pixel_size=0.5 (20x) -> overall_downsample_factor = 0.5 / 0.25 = 2.0
        self.overall_downsample_factor = self.dst_pixel_size / src_pixel_size

        # Calculate the size of the patch to read from Level 0 of the WSI
        # This is `patch_size_output` scaled back to Level 0
        # Example: patch_size_output=256, overall_downsample_factor=2.0 -> patch_size_src = 256 * 2.0 = 512
        self.patch_size_src = round(
            self.patch_size_output * self.overall_downsample_factor
        )

        # Calculate overlap in Level 0 pixels
        self.overlap_src = round(self.overlap * self.overall_downsample_factor)

        # --- Prepare backend-specific parameters (level, patch_size_at_level, overlap_at_level) ---
        # This needs to be implemented by subclasses to handle different WSI backends
        # `level` is the best pyramid level to read from.
        # `patch_size_level` is the size in pixels to read at `level`.
        # `overlap_level` is the overlap in pixels at `level`.
        self.level, self.patch_size_level, self.overlap_level = (
            self._prepare_backend_patching_params()
        )

        # --- Generate or use provided coordinates ---
        if custom_coords is None:
            self.cols, self.rows = self._calculate_grid_dimensions()
            # Generate all possible (x, y) top-left coordinates at Level 0
            all_grid_coords = np.array(
                [
                    self._grid_index_to_level0_coords(col, row)
                    for col in range(self.cols)
                    for row in range(self.rows)
                ]
            )
            self.all_coords_level0 = all_grid_coords
        else:
            if (
                not isinstance(custom_coords, np.ndarray)
                or custom_coords.ndim != 2
                or custom_coords.shape[1] != 2
            ):
                raise ValueError("custom_coords must be an (N, 2) NumPy array.")
            # Ensure custom_coords are integers (pixel coordinates)
            if not np.issubdtype(custom_coords.dtype, np.integer):
                warnings.warn("custom_coords are not integer type. Converting to int.")
                custom_coords = custom_coords.astype(np.int32)
            self.all_coords_level0 = custom_coords

        # --- Filter coordinates by mask if provided ---
        if self.mask is not None and not self.mask.empty:
            # Filter the level 0 coordinates based on the tissue mask and min_tissue_proportion
            self.valid_coords_level0 = self._filter_coords_by_mask(
                self.all_coords_level0, self.min_tissue_proportion
            )
        else:
            self.valid_coords_level0 = (
                self.all_coords_level0
            )  # All generated/custom coords are valid

    @abstractmethod
    def _prepare_backend_patching_params(self) -> Tuple[int, int, int]:
        """
        Abstract method to calculate patching parameters specific to the WSI backend.
        This includes the optimal pyramid `level` to read from, and the `patch_size`
        and `overlap` values at that specific `level`.

        Must be implemented by subclasses.

        Returns:
            Tuple[int, int, int]: (level, patch_size_at_level, overlap_at_level).
        """
        pass

    def _grid_index_to_level0_coords(self, col: int, row: int) -> Tuple[int, int]:
        """
        Converts grid indices (col, row) to their top-left pixel coordinates (x, y) at Level 0 of the WSI.
        Takes into account `patch_size_src` (size of patch to read from level 0) and `overlap_src` (overlap at level 0).
        """
        x = col * (self.patch_size_src - self.overlap_src)
        y = row * (self.patch_size_src - self.overlap_src)
        return x, y

    def _level0_coords_to_grid_index(self, x: int, y: int) -> Tuple[int, int]:
        """
        Converts Level 0 pixel coordinates (x, y) to grid indices (col, row).
        This is the inverse of `_grid_index_to_level0_coords`.
        """
        if (self.patch_size_src - self.overlap_src) == 0:
            raise ValueError(
                "Patch size minus overlap cannot be zero for grid index calculation."
            )
        col = x // (self.patch_size_src - self.overlap_src)
        row = y // (self.patch_size_src - self.overlap_src)
        return col, row

    def _calculate_grid_dimensions(self) -> Tuple[int, int]:
        """
        Calculates the number of columns and rows required to cover the WSI at Level 0,
        considering `patch_size_src` and `overlap_src`.
        """
        step_size_src = self.patch_size_src - self.overlap_src
        if step_size_src <= 0:
            raise ValueError(
                "Step size must be positive (patch_size_src > overlap_src)."
            )

        # Number of full steps + 1 if there's a remainder (to cover the last partial patch)
        cols = (
            (self.width - self.overlap_src + step_size_src - 1) // step_size_src
            if self.width > self.overlap_src
            else 1
        )
        rows = (
            (self.height - self.overlap_src + step_size_src - 1) // step_size_src
            if self.height > self.overlap_src
            else 1
        )

        # Adjust for edge cases if the slide is smaller than a single patch
        if self.width <= self.patch_size_src and self.width > 0:
            cols = 1
        if self.height <= self.patch_size_src and self.height > 0:
            rows = 1

        # If image dimension is 0, then 0 columns/rows
        if self.width == 0:
            cols = 0
        if self.height == 0:
            rows = 0

        return cols, rows

    def _filter_coords_by_mask(
        self, coords_level0: np.ndarray, min_proportion: float
    ) -> np.ndarray:
        """
        Filters the given Level 0 coordinates based on the initialized tissue `mask`
        and `min_proportion` of tissue required within each patch.

        Args:
            coords_level0 (np.ndarray): An (N, 2) array of Level 0 (x, y) coordinates.
            min_proportion (float): Minimum proportion of the patch that must be tissue.

        Returns:
            np.ndarray: Filtered (M, 2) array of Level 0 (x, y) coordinates.
        """
        if self.mask is None or self.mask.empty:
            return coords_level0  # No mask, all patches are valid

        # Convert the mask to a single shapely geometry for faster intersection checks
        # Use unary_union which is generally more efficient than repeated intersections with GeoDataFrame
        # The CRS should already be set in mask_to_gdf, but ensure it's compatible if not.
        try:
            mask_geometry = self.mask.geometry.unary_union
        except Exception as e:
            warnings.warn(
                f"Failed to create unary_union from mask geometry: {e}. Falling back to individual intersections."
            )
            mask_geometry = self.mask.geometry

        valid_indices = []
        for i, (x, y) in enumerate(coords_level0):
            # Define the patch polygon at Level 0
            patch_polygon = Polygon(
                [
                    (x, y),
                    (x + self.patch_size_src, y),
                    (x + self.patch_size_src, y + self.patch_size_src),
                    (x, y + self.patch_size_src),
                ]
            )

            if (
                not patch_polygon.is_valid
            ):  # Should ideally not happen for simple squares
                patch_polygon = shapely_make_valid(
                    patch_polygon
                )  # Use shapely's make_valid

            if not patch_polygon.is_empty:
                intersection = patch_polygon.intersection(mask_geometry)

                if not intersection.is_empty:
                    # Calculate proportion of tissue within the patch
                    # Handle GeometryCollection/MultiPolygon results from intersection
                    if (
                        intersection.geom_type == "MultiPolygon"
                        or intersection.geom_type == "GeometryCollection"
                    ):
                        # Sum areas of all valid polygons within the collection
                        intersected_area = sum(
                            p.area
                            for p in intersection.geoms
                            if p.geom_type == "Polygon" and p.is_valid
                        )
                    else:
                        intersected_area = intersection.area

                    if (
                        patch_polygon.area > 0
                        and (intersected_area / patch_polygon.area) >= min_proportion
                    ):
                        valid_indices.append(i)

        return coords_level0[valid_indices]

    def __len__(self) -> int:
        """Returns the total number of valid patches."""
        return len(self.valid_coords_level0)

    def __iter__(self) -> "WSIPatcher":
        """Returns the iterator object itself."""
        self.current_index = 0
        return self

    def __next__(self) -> Union[Tuple[np.ndarray, int, int], Tuple[int, int]]:
        """Returns the next patch or coordinate."""
        if self.current_index >= len(self):
            raise StopIteration
        item = self.__getitem__(self.current_index)
        self.current_index += 1
        return item

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[np.ndarray, int, int], Tuple[Image.Image, int, int], Tuple[int, int]
    ]:
        """Gets a patch or coordinate by its index.

        Args:
            index: The index of the valid patch coordinate.

        Returns:
            If `coords_only` is False, returns a tuple of (patch_image, x_level0, y_level0).
            If `coords_only` is True, returns a tuple of (x_level0, y_level0).

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= index < len(self):
            x_level0, y_level0 = self.valid_coords_level0[index]
            if self.coords_only:
                return x_level0, y_level0

            # Use `get_patch_at_level0_coords` to read and resize the patch
            patch_data = self.get_patch_at_level0_coords(x_level0, y_level0)
            return patch_data, x_level0, y_level0
        else:
            raise IndexError("Index out of range")

    def get_patch_at_level0_coords(
        self, x_level0: int, y_level0: int
    ) -> Union[np.ndarray, Image.Image]:
        """Reads and resizes a single patch at the given level 0 coordinates."""

        # Read the region from the WSI backend at the determined `self.level`
        # `x_level0`, `y_level0` are level 0 coordinates. These need to be scaled to `self.level`.
        # `patch_size_level` is the size to read at `self.level`.

        # Convert level 0 coordinates to coordinates at `self.level`
        x_at_level = int(round(x_level0 / self.wsi.level_downsamples[self.level]))
        y_at_level = int(round(y_level0 / self.wsi.level_downsamples[self.level]))

        patch_image_data = self.wsi.read_region(
            location=(
                x_at_level,
                y_at_level,
            ),  # Coordinates at the chosen pyramid level
            level=self.level,
            size=(
                self.patch_size_level,
                self.patch_size_level,
            ),  # Size to read at that pyramid level
            read_as=(
                "pil" if self.pil else "numpy"
            ),  # Request format based on `pil` flag
        )

        # Resize the patch to the final `self.patch_size_output` if necessary
        if self.patch_size_output != self.patch_size_level:
            if self.pil and isinstance(patch_image_data, Image.Image):
                patch_image_data = patch_image_data.resize(
                    (self.patch_size_output, self.patch_size_output),
                    Image.Resampling.LANCZOS,
                )
            elif isinstance(patch_image_data, np.ndarray):
                patch_image_data = cv2.resize(
                    patch_image_data,
                    (self.patch_size_output, self.patch_size_output),
                    interpolation=cv2.INTER_AREA,
                )  # INTER_AREA for shrinking
                if (
                    patch_image_data.ndim == 2
                ):  # Convert grayscale to 3 channels if it somehow becomes 2D
                    patch_image_data = np.stack([patch_image_data] * 3, axis=-1)
                elif patch_image_data.shape[-1] == 4:  # Drop alpha channel if present
                    patch_image_data = patch_image_data[:, :, :3]

        return patch_image_data

    def get_cols_rows(self) -> Tuple[int, int]:
        """Get the number of columns and rows in the patch grid."""
        # This function is only meaningful if custom_coords was NOT used.
        if self.custom_coords is not None:
            warnings.warn(
                "`get_cols_rows` is not meaningful when `custom_coords` are used."
            )
            return (0, 0)
        return self.cols, self.rows

    def visualize(self, target_width: int = 1000) -> Image.Image:
        """
        Overlays patch coordinates computed by the WSIPatcher onto a scaled thumbnail of the WSI.
        It creates a visualization of the patcher coordinates and returns it as a PIL Image.

        Args:
            target_width (int): Desired width of the visualization thumbnail.

        Returns
        -------
        Image.Image
            Patch visualization
        """
        # Ensure WSI object is fully loaded
        self.wsi._lazy_initialize()

        # Calculate thumbnail dimensions and scale factor
        wsi_width, wsi_height = self.wsi.get_dimensions()
        downsample_factor = target_width / wsi_width if wsi_width > 0 else 1.0
        thumbnail_width = int(wsi_width * downsample_factor)
        thumbnail_height = int(wsi_height * downsample_factor)

        # Get thumbnail as NumPy array for OpenCV drawing
        canvas_np = np.array(
            self.wsi.get_thumbnail((thumbnail_width, thumbnail_height))
        ).astype(np.uint8)

        # Calculate patch size at thumbnail resolution for drawing rectangles
        # patch_size_src is Level 0 patch size. Scale it down by downsample_factor
        thumbnail_patch_size = max(1, int(self.patch_size_src * downsample_factor))
        thickness = max(1, thumbnail_patch_size // 50)  # Dynamic thickness

        # Draw patches (valid_coords_level0 holds the Level 0 coordinates)
        for x_lvl0, y_lvl0 in self.valid_coords_level0:
            x_thumb = int(x_lvl0 * downsample_factor)
            y_thumb = int(y_lvl0 * downsample_factor)

            # Draw rectangle with red color
            cv2.rectangle(
                canvas_np,
                (x_thumb, y_thumb),
                (x_thumb + thumbnail_patch_size, y_thumb + thumbnail_patch_size),
                (0, 0, 255),  # BGR for red
                thickness,
            )

        # Overlay tissue contours if available
        if self.mask is not None and not self.mask.empty:
            for geom in self.mask.geometry:
                if geom.is_empty:
                    continue

                # Scale coordinates to thumbnail size
                exterior_coords = np.array(geom.exterior.coords) * downsample_factor
                exterior_coords = exterior_coords.astype(np.int32)

                # Draw polygon
                cv2.polylines(
                    canvas_np,
                    [exterior_coords],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

                # Draw holes
                for interior in geom.interiors:
                    interior_coords = np.array(interior.coords) * downsample_factor
                    interior_coords = interior_coords.astype(np.int32)
                    cv2.polylines(
                        canvas_np,
                        [interior_coords],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )

        # Add informative text (optional)
        text_area_height = 130  # For text info area
        text_offset_x = int(thumbnail_width * 0.02)  # 2% from left
        text_spacing_y = 25  # Vertical spacing

        # Create a semi-transparent background for text if desired
        # canvas_np[0:text_area_height, 0:min(thumbnail_width, 350)] = (canvas_np[0:text_area_height, 0:min(thumbnail_width, 350)] * 0.7).astype(np.uint8)

        # Convert to BGR for text drawing
        if (
            canvas_np.shape[-1] == 3 and canvas_np[0, 0, 0] > canvas_np[0, 0, 2]
        ):  # Simple check for RGB vs BGR
            canvas_np = cv2.cvtColor(canvas_np, cv2.COLOR_RGB2BGR)

        cv2.putText(
            canvas_np,
            f"Patches: {len(self)}",
            (text_offset_x, text_spacing_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas_np,
            f"WSI Dims: {wsi_width}x{wsi_height}",
            (text_offset_x, text_spacing_y * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas_np,
            f"WSI MPP: {self.wsi.mpp:.2f} um/px",
            (text_offset_x, text_spacing_y * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas_np,
            f"WSI Mag: {self.wsi.mag}x",
            (text_offset_x, text_spacing_y * 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas_np,
            f"Patch: {self.patch_size_output}px @ {self.dst_mag}x ({self.overlap}ovlp)",
            (text_offset_x, text_spacing_y * 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Convert back to RGB for PIL Image
        canvas_np = cv2.cvtColor(canvas_np, cv2.COLOR_BGR2RGB)
        return Image.fromarray(canvas_np)


class OpenSlideWSIPatcher(WSIPatcher):
    """A WSIPatcher implementation for OpenSlide-backed WSIs."""

    def _prepare_backend_patching_params(self) -> Tuple[int, int, int]:
        """
        Calculates patching parameters specific to OpenSlide.
        Determines the optimal OpenSlide pyramid level to read from.
        """
        # `overall_downsample_factor` is the total downsample from Level 0 to target output resolution.
        # OpenSlide's `get_best_level_for_downsample` will find the best level that is *at least*
        # as high resolution as needed.
        level, custom_downsample_at_level = (
            self.wsi.get_best_level_and_custom_downsample(
                self.overall_downsample_factor
            )
        )

        # `level_downsample` is the actual downsample factor of the chosen `level` relative to level 0.
        level_downsample = self.wsi.level_downsamples[level]

        # `patch_size_at_level` is the size in pixels to read from the chosen `level`.
        # This is `patch_size_output` scaled by `custom_downsample_at_level` (which means if custom_downsample_at_level is 2.0,
        # it means current level is 2x lower resolution than needed for patch_size_output, so we need to read a 2x larger patch from this level).
        # OR more simply: patch_size_at_level = (patch_size_output * overall_downsample_factor) / actual_level_downsample
        # which simplifies to patch_size_output * custom_downsample_at_level
        patch_size_level = round(self.patch_size_output * custom_downsample_at_level)

        # `overlap_at_level` is the overlap in pixels at the chosen `level`.
        overlap_level = round(self.overlap * custom_downsample_at_level)

        return level, patch_size_level, overlap_level


class CuImageWSIPatcher(WSIPatcher):
    """A WSIPatcher implementation for cuCIM-backed WSIs."""

    def _prepare_backend_patching_params(self) -> Tuple[int, int, int]:
        """
        Calculates patching parameters specific to cuCIM.
        Determines the optimal cuCIM pyramid level to read from.
        """
        # `overall_downsample_factor` is the total downsample from Level 0 to target output resolution.
        level, custom_downsample_at_level = (
            self.wsi.get_best_level_and_custom_downsample(
                self.overall_downsample_factor
            )
        )

        # `level_downsample` is the actual downsample factor of the chosen `level` relative to level 0.
        level_downsample = self.wsi.level_downsamples[level]

        # `patch_size_at_level` is the size in pixels to read from the chosen `level`.
        patch_size_level = round(self.patch_size_output * custom_downsample_at_level)

        # `overlap_at_level` is the overlap in pixels at the chosen `level`.
        overlap_level = round(self.overlap * custom_downsample_at_level)

        return level, patch_size_level, overlap_level


class NumpyWSIPatcher(WSIPatcher):
    """A WSIPatcher implementation for NumPy array-backed WSIs."""

    def _prepare_backend_patching_params(self) -> Tuple[int, int, int]:
        """
        Sets patching parameters for a NumPy array.
        Since NumPy arrays are single-resolution, level is set to 0 (effectively level 0).
        `patch_size_level` and `overlap_level` are directly scaled versions from Level 0 to input for model.
        """
        # For a NumPy array, there's effectively only one level, which we call 0.
        level = 0

        # The `overall_downsample_factor` applies directly here.
        # patch_size_level is the size of the region to read from the NumPy array
        # to later resize to `self.patch_size_output`.
        # Example: patch_size_output=256, overall_downsample_factor=2 (for 40x->20x)
        # patch_size_level = round(256 * 2) = 512.
        patch_size_level = round(
            self.patch_size_output * self.overall_downsample_factor
        )

        overlap_level = round(self.overlap * self.overall_downsample_factor)

        # Special handling: if patch_size_level is 0, make it 1 to avoid division by zero errors or invalid dimensions.
        if patch_size_level == 0:
            warnings.warn(
                "Calculated patch_size_level is 0. Setting to 1 to prevent errors. Check input patch_size and scaling factors."
            )
            patch_size_level = 1

        return level, patch_size_level, overlap_level
