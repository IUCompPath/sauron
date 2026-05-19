from typing import Union

import cv2
import geopandas as gpd
import numpy as np
import openslide
from PIL import Image

from aegis.utils.WSIObjects import WholeSlideImage, wsi_factory


def draw_contours(
    image: np.ndarray,
    contours: gpd.GeoDataFrame,
    draw_outline: bool = False,
    line_thickness: int = 1,
    scale_factor: float = 1.0,
    contour_color: tuple = (0, 255, 0),
) -> np.ndarray:
    """
    Draw contours on an image.

    Args:
        image (np.ndarray): Image on which to draw.
        contours (gpd.GeoDataFrame): Contours to draw.
        draw_outline (bool): Whether to draw the outline of contours.
        line_thickness (int): Thickness of the contour lines.
        scale_factor (float): Scaling factor for the contours.
        contour_color (tuple): Color of the contours.

    Returns:
        np.ndarray: Image with contours drawn.
    """
    for _, group in contours.groupby("tissue_id"):
        for _, row in group.iterrows():
            exterior = np.array(
                [
                    [
                        int(round(x * scale_factor)),
                        int(round(y * scale_factor)),
                    ]
                    for x, y in row.geometry.exterior.coords
                ]
            )
            interiors = [
                np.array(
                    [
                        [
                            int(round(x * scale_factor)),
                            int(round(y * scale_factor)),
                        ]
                        for x, y in interior.coords
                    ]
                )
                for interior in row.geometry.interiors
            ]

            cv2.drawContours(
                image,
                [exterior],
                contourIdx=-1,
                color=contour_color,
                thickness=cv2.FILLED,
            )
            for hole in interiors:
                cv2.drawContours(
                    image,
                    [hole],
                    contourIdx=-1,
                    color=(0, 0, 0),
                    thickness=cv2.FILLED,
                )
            if draw_outline:
                cv2.drawContours(
                    image,
                    [exterior],
                    contourIdx=-1,
                    color=contour_color,
                    thickness=line_thickness,
                )
    return image


def visualize_tissue(
    wsi: Union[np.ndarray, openslide.OpenSlide, WholeSlideImage],
    tissue_contours: gpd.GeoDataFrame,
    contour_color: tuple = (0, 255, 0),
    line_thickness: int = 5,
    target_width: int = 1000,
) -> Image.Image:
    """
    Visualize tissue contours on a whole slide image.

    Args:
        wsi (Union[np.ndarray, openslide.OpenSlide, WSI]): The whole slide image.
        tissue_contours (gpd.GeoDataFrame): Contours of the tissue regions.
        contour_color (tuple): Color of the contour lines.
        line_thickness (int): Thickness of the contour lines.
        target_width (int): Target width for the visualization image.

    Returns:
        Image.Image: The visualization image.
    """
    wsi = wsi_factory(wsi)
    width, height = wsi.get_dimensions()
    scale_factor = target_width / width

    thumbnail = wsi.get_thumbnail(
        width=int(width * scale_factor), height=int(height * scale_factor)
    )

    if tissue_contours.empty:
        return Image.fromarray(thumbnail)

    overlay = np.zeros_like(thumbnail, dtype=np.uint8)
    overlay = draw_contours(
        overlay,
        tissue_contours,
        draw_outline=True,
        line_thickness=line_thickness,
        scale_factor=scale_factor,
        contour_color=contour_color,
    )

    alpha = 0.4
    blended = cv2.addWeighted(thumbnail, 1 - alpha, overlay, alpha, 0)
    return Image.fromarray(blended)
