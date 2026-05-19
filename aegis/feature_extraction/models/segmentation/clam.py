import torch
from torchvision import transforms

from aegis.feature_extraction.models.segmentation.classic_segmenter_utils import (
    tissue_mask,
)
from aegis.feature_extraction.models.segmentation.factory import (
    SegmentationModel,
)


class ClamSegmenter(SegmentationModel):
    def __init__(self, **build_kwargs):
        """
        ClamSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        """
        Build and load ClamSegmenter model.
        """
        # Clam segmenter does not use a PyTorch model, so model is None
        model = None
        # No specific transforms needed as it operates on numpy arrays
        eval_transforms = transforms.Compose([])
        # Input size is not fixed for classic segmenter, but we need to provide a value
        input_size = 0
        # Precision is not applicable for classic segmenter
        precision = torch.float32
        # Target magnification can be set to a default or configured
        target_mag = 1  # Operates on the whole slide or a downsampled version

        return model, eval_transforms, input_size, precision, target_mag

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Custom forward pass for ClamSegmenter.
        """

        # Convert torch tensor to numpy array
        # Assuming image is (batch_size, C, H, W) and we only process one image at a time (batch_size=1)
        # Also assuming C=3 (RGB)
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Apply the tissue_mask function
        mask_np = tissue_mask(
            image_np,
            sthresh=20,
            mthresh=7,
            close=4,
            use_otsu=False,
            a_t=100,
            a_h=16,
            max_n_holes=8,
        )

        # Convert numpy mask back to torch tensor
        # Ensure it's a binary mask (0 or 1) and correct dtype
        predictions = (
            torch.from_numpy(mask_np).unsqueeze(0).to(torch.uint8)
        )  # Add batch dimension back

        return predictions
