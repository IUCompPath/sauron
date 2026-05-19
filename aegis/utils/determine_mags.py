import logging
import os
import re
from pathlib import Path
from typing import NewType
from xml.dom import minidom

import openslide
import pandas as pd

SlideMPP = NewType("SlideMPP", float)
_logger = logging.getLogger("aegis")


def _extract_mpp_from_comments(slide: openslide.AbstractSlide) -> SlideMPP | None:
    slide_properties = slide.properties.get("openslide.comment", "")
    pattern = r"<PixelSizeMicrons>(.*?)</PixelSizeMicrons>"
    match = re.search(pattern, slide_properties)
    if match is not None and (mpp := match.group(1)) is not None:
        return SlideMPP(float(mpp))
    else:
        return None


class MPPExtractionError(Exception):
    """Raised when the Microns Per Pixel (MPP) extraction from the slide's metadata fails"""

    pass


def _extract_mpp_from_metadata(slide: openslide.AbstractSlide) -> SlideMPP | None:
    try:
        xml_path = slide.properties.get("tiff.ImageDescription") or None
        if xml_path is None:
            return None
        doc = minidom.parseString(xml_path)
        collection = doc.documentElement
        images = collection.getElementsByTagName("Image")
        pixels = images[0].getElementsByTagName("Pixels")
        mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
    except Exception:
        _logger.exception("failed to extract MPP from image description")
        return None
    return SlideMPP(mpp)


def get_slide_mpp_(
    slide: openslide.AbstractSlide | Path, *, default_mpp: SlideMPP | None
) -> SlideMPP | None:
    """
    Retrieve the microns per pixel (MPP) value from a slide.
    This function attempts to extract the MPP value from the given slide. If the slide
    is provided as a file path, it will be opened using OpenSlide. The function first
    checks for the MPP value in the slide's properties. If not found, it attempts to
    extract the MPP value from the slide's comments and metadata. If all attempts fail
    and a default MPP value is provided, it will use the default value. If no MPP value
    can be determined and no default is provided, an MPPExtractionError is raised.
    Args:
        slide: The slide object or file path to the slide.
        default_mpp: The default MPP value to use if extraction fails.
    Returns:
        The extracted or default MPP value, or None if extraction fails and no default is provided.
    Raises:
        MPPExtractionError: If the MPP value cannot be determined and no default is provided.
    """

    if isinstance(slide, Path):
        slide = openslide.open_slide(slide)

    if openslide.PROPERTY_NAME_MPP_X in slide.properties:
        slide_mpp = SlideMPP(float(slide.properties[openslide.PROPERTY_NAME_MPP_X]))
    elif slide_mpp := _extract_mpp_from_comments(slide):
        pass
    elif slide_mpp := _extract_mpp_from_metadata(slide):
        pass

    if slide_mpp is None and default_mpp:
        _logger.warning(
            f"could not infer slide MPP from metadata, using {default_mpp} instead."
        )
    elif slide_mpp is None and default_mpp is None:
        raise MPPExtractionError()

    return slide_mpp or default_mpp


subtypes = os.listdir("/N/project/hancock/Scripts/TCGA-COHORT")

info_df = pd.DataFrame(
    columns=["subtype", "patient", "slide_file", "slide_mpp", "slide_path"]
)

for subtype in subtypes:
    for patient in os.listdir(f"/N/project/hancock/Scripts/TCGA-COHORT/{subtype}"):
        print(f"Processing {subtype}/{patient}")
        for slide_file in os.listdir(
            f"/N/project/hancock/Scripts/TCGA-COHORT/{subtype}/{patient}"
        ):
            slide_path = Path(
                f"/N/project/hancock/Scripts/TCGA-COHORT/{subtype}/{patient}/{slide_file}"
            )
            slide_mpp = get_slide_mpp_(slide_path, default_mpp=None)
            new_row = pd.DataFrame(
                [
                    {
                        "subtype": subtype,
                        "patient": patient,
                        "slide_file": slide_file,
                        "slide_mpp": slide_mpp,
                        "slide_path": slide_path,
                    }
                ]
            )
            info_df = pd.concat([info_df, new_row], ignore_index=True)

info_df.to_csv(
    "/N/project/hancock/Scripts/TCGA-COHORT/magnification_report.csv", index=False
)
