from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from aegis.feature_extraction.wsi.patching import WSIPatcher


class WSIPatcherDataset(Dataset):
    """Dataset from a WSI patcher to directly read tiles on a slide"""

    def __init__(self, patcher: WSIPatcher, transform: Optional[Callable] = None):
        """
        Initializes the dataset.

        Args:
            patcher (WSIPatcher): An instance of WSIPatcher configured to yield image patches.
            transform (Optional[Callable]): A callable transform to apply to each patch (e.g., torchvision.transforms).
        """
        self.patcher = patcher
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patcher)

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[torch.Tensor, Tuple[int, int]], Tuple[Tuple[int, int]]]:
        """
        Retrieves a patch and its coordinates.

        Args:
            index (int): The index of the patch to retrieve.

        Returns:
            Union[Tuple[torch.Tensor, Tuple[int, int]], Tuple[Tuple[int, int]]]:
                If patcher is `coords_only=False`: (transformed_patch_tensor, (x_coord_level0, y_coord_level0)).
                If patcher is `coords_only=True`: ((x_coord_level0, y_coord_level0)).
        """
        # patcher[index] will return (image_data, x, y) or (x, y) based on coords_only flag
        item = self.patcher[index]

        if self.patcher.coords_only:
            # item is (x, y)
            return item
        else:
            # item is (image_data, x, y)
            image_data, x, y = item

            if self.transform:
                # Ensure image_data is PIL Image if transform expects it, or numpy if transform handles that
                if isinstance(image_data, np.ndarray):
                    image_data = Image.fromarray(
                        image_data
                    )  # Convert to PIL for common torchvision transforms
                transformed_image = self.transform(image_data)
                return transformed_image, (x, y)
            else:
                # If no transform, return original image data and coords
                # Convert to tensor if it's numpy array and no transform is applied, for consistency with ML pipelines
                if isinstance(image_data, np.ndarray):
                    # Ensure channels-first (CxHxW) for PyTorch if no transform is applied
                    if image_data.ndim == 3 and image_data.shape[2] == 3:  # HWC
                        image_data = np.transpose(image_data, (2, 0, 1))  # CHW
                    return torch.from_numpy(image_data).float(), (x, y)
                return image_data, (x, y)  # Return as is if already PIL or other format
