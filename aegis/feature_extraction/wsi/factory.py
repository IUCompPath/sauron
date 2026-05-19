import os
from typing import Any, Literal, Optional

import openslide
from PIL import Image

# Import base WSI class for type hinting
from aegis.feature_extraction.wsi.base import WSI
from aegis.feature_extraction.wsi.image import ImageWSI
from aegis.feature_extraction.wsi.openslide import OpenSlideWSI

# Conditional import for CuCIM to handle optional dependency
try:
    from aegis.feature_extraction.wsi.cucim import CuCIMWSI

    _CUCIM_AVAILABLE = True
except ImportError:
    _CUCIM_AVAILABLE = False
    import warnings

    from aegis.feature_extraction.utils.io import CuImageWarning

    warnings.simplefilter("once", CuImageWarning)  # Warn only once


# Define common WSI extensions and their typical readers
WSIReaderType = Literal["openslide", "image", "cucim"]
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
}  # .czi added as OpenSlide supports it from 4.0.0
CUCIM_EXTENSIONS = {".svs", ".tif", ".tiff", ".czi", ".ndpi"}  # CuCIM supports these
PIL_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}  # PIL supports these


def load_wsi(
    slide_path: str,
    reader_type: Optional[WSIReaderType] = None,
    **kwargs: Any,  # Capture additional args for WSI constructor
) -> WSI:
    """
    Load a whole-slide image (WSI) using the appropriate backend.

    By default, uses OpenSlideWSI for OpenSlide-supported file extensions,
    and ImageWSI for others. Users may override this behavior by explicitly
    specifying a reader using the `reader_type` argument.

    Parameters
    ----------
    slide_path : str
        Path to the whole-slide image.
    reader_type : {'openslide', 'image', 'cucim'}, optional
        Manually specify the WSI reader to use. If None (default), selection
        is automatic based on file extension.
    **kwargs : Any
        Additional keyword arguments passed to the WSI reader constructor.
        Common kwargs: `original_path`, `name`, `tissue_seg_path`, `custom_mpp_keys`, `lazy_init`, `mpp`, `max_workers`.

    Returns
    -------
    WSI
        An instance of the appropriate WSI reader subclass.

    Raises
    ------
    ValueError
        If `reader_type` is 'cucim' but the cucim package is not installed.
        Or if an unknown reader type is specified.
    FileNotFoundError
        If the slide_path does not exist.
    """
    if not os.path.exists(slide_path):
        raise FileNotFoundError(f"WSI file not found at: {slide_path}")

    ext = os.path.splitext(slide_path)[1].lower()

    # Try to open image source object first, as some WSI classes accept already opened objects
    # This avoids redundant file opening by downstream WSI constructors
    image_source_obj: Any = None
    if reader_type == "openslide" or (
        reader_type is None and ext in OPENSLIDE_EXTENSIONS
    ):
        try:
            image_source_obj = openslide.OpenSlide(slide_path)
            # Ensure it's not a broken or empty OpenSlide handle
            if image_source_obj.dimensions == (0, 0):
                raise openslide.OpenSlideError(
                    f"OpenSlide returned 0x0 dimensions for {slide_path}. Likely corrupted or unsupported format."
                )
        except openslide.OpenSlideError as e:
            # If OpenSlide fails, print warning and try next reader if reader_type was None
            print(
                f"Warning: OpenSlide failed to open {slide_path}: {e}. Trying other readers if reader_type is auto."
            )
            if (
                reader_type == "openslide"
            ):  # If explicitly requested OpenSlide, re-raise
                raise e
            reader_type = None  # Reset to auto-select

    if _CUCIM_AVAILABLE and (
        reader_type == "cucim"
        or (
            reader_type is None and ext in CUCIM_EXTENSIONS and image_source_obj is None
        )
    ):
        try:
            from cucim import CuImage

            image_source_obj = CuImage(slide_path)
        except Exception as e:
            print(
                f"Warning: CuCIM failed to open {slide_path}: {e}. Trying other readers if reader_type is auto."
            )
            if reader_type == "cucim":  # If explicitly requested CuCIM, re-raise
                raise e
            reader_type = None  # Reset to auto-select

    if reader_type == "image" or (reader_type is None and image_source_obj is None):
        # If no other reader succeeded or explicitly image reader
        try:
            image_source_obj = Image.open(slide_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to open {slide_path} with PIL.Image: {e}. No suitable reader found."
            )

    # Now instantiate the WSI class with the opened image_source_obj
    if reader_type == "openslide":
        if not isinstance(image_source_obj, openslide.OpenSlide):
            raise TypeError(
                f"Expected openslide.OpenSlide object, but got {type(image_source_obj)} after opening {slide_path}. Check file format compatibility."
            )
        return OpenSlideWSI(
            slide_path=slide_path, image_source=image_source_obj, **kwargs
        )

    elif reader_type == "image":
        if not isinstance(image_source_obj, Image.Image):
            raise TypeError(
                f"Expected PIL.Image.Image object, but got {type(image_source_obj)} after opening {slide_path}. Check file format compatibility."
            )
        return ImageWSI(image_source=image_source_obj, slide_path=slide_path, **kwargs)

    elif reader_type == "cucim":
        if not _CUCIM_AVAILABLE:
            raise ValueError(
                f"Unsupported file format '{ext}' for CuCIM. Or CuCIM is not installed. "
                f"Supported formats: {', '.join(CUCIM_EXTENSIONS)}."
            )
        # image_source_obj should be CuImage if we reach here via CuCIM path
        return CuCIMWSI(image_source=image_source_obj, slide_path=slide_path, **kwargs)

    elif reader_type is None:  # Auto-detection based on successful opening
        if isinstance(image_source_obj, openslide.OpenSlide):
            return OpenSlideWSI(
                slide_path=slide_path, image_source=image_source_obj, **kwargs
            )
        elif _CUCIM_AVAILABLE and isinstance(image_source_obj, CuImage):
            return CuCIMWSI(
                image_source=image_source_obj, slide_path=slide_path, **kwargs
            )
        elif isinstance(image_source_obj, Image.Image):
            return ImageWSI(
                image_source=image_source_obj, slide_path=slide_path, **kwargs
            )
        else:
            raise RuntimeError(
                f"Failed to auto-determine WSI reader for {slide_path}. No suitable backend could open the file."
            )

    else:
        raise ValueError(
            f"Unknown reader_type: {reader_type}. Choose from 'openslide', 'image', or 'cucim'."
        )
