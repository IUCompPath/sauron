import os
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from aegis.feature_extraction.models.segmentation.classic_segmenter_utils import (
    tissue_mask,
)
from aegis.feature_extraction.utils.io import (
    get_dir,
    get_weights_path,
    has_internet_connection,
)


class SegmentationModel(torch.nn.Module):
    _has_internet = has_internet_connection()
    # Added max_holes_to_fill as a class attribute for models that use it
    max_holes_to_fill: int = 20  # Default, can be overridden by subclasses or factory.

    def __init__(self, freeze=True, confidence_thresh=0.5, **build_kwargs):
        """
        Initialize Segmentation model wrapper.

        Args:
            freeze (bool, optional): If True, the model's parameters are frozen
                (i.e., not trainable) and the model is set to evaluation mode.
                Defaults to True.
            confidence_thresh (float, optional): Threshold for prediction confidence.
                Predictions below this threshold may be filtered out or ignored.
                Default is 0.5. Set to 0.4 to keep more tissue.
            **build_kwargs: Additional keyword arguments passed to the internal
                `_build` method.

        Attributes:
            model (torch.nn.Module): The constructed model.
            eval_transforms (Callable): Transformations to apply to input data during inference.
            input_size (int): Expected input image size for the model.
            precision (torch.dtype): Recommended precision for model inference (e.g., torch.float16).
            target_mag (int): The magnification level at which the model is intended to operate (e.g., 10x).
        """
        super().__init__()
        (
            self.model,
            self.eval_transforms,
            self.input_size,
            self.precision,
            self.target_mag,
        ) = self._build(**build_kwargs)
        self.confidence_thresh = confidence_thresh
        self.model_name = self.__class__.__name__  # Store model name for config

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, image):
        """
        Can be overwritten if model requires special forward pass.
        """
        z = self.model(image)
        return z

    @abstractmethod
    def _build(
        self, **build_kwargs
    ) -> tuple[nn.Module, transforms.Compose, int, torch.dtype, int]:
        """
        Build the segmentation model and preprocessing transforms.
        Returns: model, eval_transforms, input_size, precision, target_mag
        """
        pass


class HESTSegmenter(SegmentationModel):
    def __init__(self, **build_kwargs):
        """
        HESTSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        """
        Build and load HESTSegmenter model.

        Returns:
            Tuple[nn.Module, transforms.Compose, int, torch.dtype, int]: Model, transforms, input size, precision, target magnification.
        """

        from torchvision.models.segmentation import deeplabv3_resnet50

        model_ckpt_name = "deeplabv3_seg_v4.ckpt"
        weights_path = get_weights_path("seg", "hest")

        # Check if a path is provided but doesn't exist
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Expected checkpoint at '{weights_path}', but the file was not found."
            )

        # Initialize base model
        model = deeplabv3_resnet50(weights=None)
        model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1, stride=1)

        if not weights_path:
            if not SegmentationModel._has_internet:
                raise FileNotFoundError(
                    f"Internet connection not available and checkpoint not found locally in model registry at aegis/feature_extraction/models/segmentation/checkpoints.json.\n\n"
                    f"To proceed, please manually download {model_ckpt_name} from:\n"
                    f"https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    f"and place it at:\ncheckpoints.json"  # refers to the local_ckpts.json in this directory
                )

            # If internet is available, download from HuggingFace
            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type="model",
                local_dir=get_dir(),  # Cache directory resolved by get_dir()
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name],
            )

            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        # Load and clean checkpoint
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = {
            k.replace("model.", ""): v
            for k, v in checkpoint.get("state_dict", {}).items()
            if "aux" not in k
        }

        model.load_state_dict(state_dict)

        # Store configuration
        input_size = 512
        precision = torch.float16
        target_mag = 10

        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        return model, eval_transforms, input_size, precision, target_mag

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # input should be of shape (batch_size, C, H, W)
        assert (
            len(image.shape) == 4
        ), f"Input must be 4D image tensor (shape: batch_size, C, H, W), got {image.shape} instead"
        logits = self.model(image)["out"]
        softmax_output = F.softmax(logits, dim=1)
        predictions = (softmax_output[:, 1, :, :] > self.confidence_thresh).to(
            torch.uint8
        )  # Shape: [bs, 512, 512]
        return predictions


class JpegCompressionTransform:
    def __init__(self, quality=80):
        self.quality = quality

    def __call__(self, image):
        import cv2
        from PIL import Image

        # Convert PIL Image to NumPy array
        image = np.array(image)

        # Apply JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, image = cv2.imencode(".jpg", image, encode_param)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Convert back to PIL Image
        return Image.fromarray(image)


class GrandQCArtifactSegmenter(SegmentationModel):
    _class_mapping = {
        1: "Normal Tissue",
        2: "Fold",
        3: "Darkspot & Foreign Object",
        4: "PenMarking",
        5: "Edge & Air Bubble",
        6: "OOF",
        7: "Background",
    }

    def __init__(self, **build_kwargs):
        """
        GrandQCArtifactSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, remove_penmarks_only=False):
        """
        Load the GrandQC artifact removal segmentation model.
        Credit: https://www.nature.com/articles/s41467-024-54769-y
        """

        import segmentation_models_pytorch as smp

        self.remove_penmarks_only = (
            remove_penmarks_only  # ignore all other artifacts than penmakrs.
        )
        model_ckpt_name = "GrandQC_MPP1_state_dict.pth"
        encoder_name = "timm-efficientnet-b0"
        encoder_weights = "imagenet"
        weights_path = get_weights_path("seg", "grandqc_artifact")

        # Verify that user-provided weights_path is valid
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Expected checkpoint at '{weights_path}', but the file was not found."
            )

        # Attempt to download if file is missing and not already available
        if not weights_path:
            if not SegmentationModel._has_internet:
                raise FileNotFoundError(
                    f"Internet connection not available and checkpoint not found locally.\n\n"
                    f"To proceed, please manually download {model_ckpt_name} from:\n"
                    f"https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    f"and place it at:\ncheckpoints.json"  # refers to the local_ckpts.json in this directory
                )

            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type="model",
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name],
            )

            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        # Initialize model
        # BUG FIX: Assign the created model instance to a local variable `model_instance`
        # and then use it. The `model` variable was not initialized before this point.
        model_instance = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=8,  # GrandQC artifact model has 8 output classes
            activation=None,
        )

        # Load checkpoint
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model_instance.load_state_dict(state_dict)  # Use the initialized instance

        # Model config
        input_size = 512
        precision = torch.float32
        target_mag = 10  # This model is often used on 10x or 1x (MPP1)
        # Add a default for max_holes_to_fill if this segmenter needs it
        self.max_holes_to_fill = 20  # Can be configured per model type

        # Evaluation transforms
        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # BUG FIX: Return the initialized `model_instance`
        return model_instance, eval_transforms, input_size, precision, target_mag

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Custom forward pass.
        """
        logits = self.model.predict(image)
        probs = torch.softmax(logits, dim=1)
        _, predicted_classes = torch.max(probs, dim=1)
        if self.remove_penmarks_only:
            # Class 4 is PenMarking, Class 7 is Background (from GrandQC paper)
            predictions = torch.where(
                (predicted_classes == 4) | (predicted_classes == 7), 0, 1
            )
        else:
            # Classes > 1 typically represent artifacts or background.
            # Assuming GrandQC artifact model maps normal tissue to 1.
            predictions = torch.where(predicted_classes > 1, 0, 1)
        predictions = predictions.to(torch.uint8)

        return predictions


class GrandQCSegmenter(SegmentationModel):
    def __init__(self, **build_kwargs):
        """
        GrandQCSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        """
        Load the GrandQC tissue detection segmentation model.
        Credit: https://www.nature.com/articles/s41467-024-54769-y
        """
        import segmentation_models_pytorch as smp

        model_ckpt_name = "Tissue_Detection_MPP10.pth"
        encoder_name = "timm-efficientnet-b0"
        encoder_weights = "imagenet"
        weights_path = get_weights_path("seg", "grandqc")

        # Verify that user-provided weights_path is valid
        if weights_path and not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Expected checkpoint at '{weights_path}', but the file was not found."
            )

        # Verify checkpoint path
        if not weights_path:
            if not SegmentationModel._has_internet:
                raise FileNotFoundError(
                    f"Internet connection not available and checkpoint not found locally at '{weights_path}'.\n\n"
                    f"To proceed, please manually download {model_ckpt_name} from:\n"
                    f"https://huggingface.co/MahmoodLab/hest-tissue-seg/\n"
                    f"and place it at:\ncheckpoints.json"  # refers to the local_ckpts.json in this directory
                )

            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(
                repo_id="MahmoodLab/hest-tissue-seg",
                repo_type="model",
                local_dir=get_dir(),
                cache_dir=get_dir(),
                allow_patterns=[model_ckpt_name],
            )
            weights_path = os.path.join(checkpoint_dir, model_ckpt_name)

        # Initialize model
        # BUG FIX: Assign the created model instance to a local variable `model_instance`
        # and then use it. The `model` variable was not initialized before this point.
        model_instance = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=2,  # GrandQC tissue detection has 2 output classes (tissue/background)
            activation=None,
        )

        # Load checkpoint
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model_instance.load_state_dict(state_dict)  # Use the initialized instance

        # Model config
        input_size = 512
        precision = torch.float32
        target_mag = 10  # GrandQC tissue detection is typically for 10x or 1x (MPP1)
        # Add a default for max_holes_to_fill if this segmenter needs it
        self.max_holes_to_fill = 20  # Can be configured per model type

        # Evaluation transforms
        eval_transforms = transforms.Compose(
            [
                JpegCompressionTransform(quality=80),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # BUG FIX: Return the initialized `model_instance`
        return model_instance, eval_transforms, input_size, precision, target_mag

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Custom forward pass.
        """
        logits = self.model.predict(image)
        probs = torch.softmax(logits, dim=1)
        max_probs, predicted_classes = torch.max(probs, dim=1)
        # In GrandQC, class 0 is tissue, class 1 is background.
        # We want to keep class 0 (tissue) if its probability is above threshold.
        # So, if max_probs >= threshold AND predicted_class == 0.
        # (1 - predicted_classes) converts 0 to 1 and 1 to 0. So, we want predictions == 1.
        predictions = (max_probs >= self.confidence_thresh) * (1 - predicted_classes)
        predictions = predictions.to(torch.uint8)

        return predictions


class ClassicSegmenter(SegmentationModel):
    def __init__(self, **build_kwargs):
        """
        ClassicSegmenter initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self):
        """
        Build and load ClassicSegmenter model.
        """
        # Classic segmenter does not use a PyTorch model, so model is None
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
        Custom forward pass for ClassicSegmenter.
        """

        # Convert torch tensor to numpy array
        # Assuming image is (batch_size, C, H, W) and we only process one image at a time (batch_size=1)
        # Also assuming C=3 (RGB)
        image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Apply the tissue_mask function
        mask_np = tissue_mask(image_np)

        # Convert numpy mask back to torch tensor
        # Ensure it's a binary mask (0 or 1) and correct dtype
        predictions = (
            torch.from_numpy(mask_np).unsqueeze(0).to(torch.uint8)
        )  # Add batch dimension back

        return predictions


def segmentation_model_factory(
    model_name: str,
    confidence_thresh: float = 0.5,
    freeze: bool = True,
    **build_kwargs,
) -> SegmentationModel:
    """
    Factory function to build a segmentation model by name.
    """

    if "device" in build_kwargs:
        import warnings

        warnings.warn(
            "Passing `device` to `segmentation_model_factory` is deprecated as of version 0.1.0 "
            "Please pass `device` when segmenting the tissue, e.g., `slide.segment_tissue(..., device='cuda:0')`.",
            DeprecationWarning,
            stacklevel=2,
        )

    if model_name == "hest":
        return HESTSegmenter(
            freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs
        )
    elif model_name == "grandqc":
        return GrandQCSegmenter(
            freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs
        )
    elif model_name == "grandqc_artifact":
        return GrandQCArtifactSegmenter(
            freeze=freeze, confidence_thresh=confidence_thresh, **build_kwargs
        )
    elif model_name == "classic":
        return ClassicSegmenter(freeze=freeze, **build_kwargs)
    elif model_name == "clam":
        from aegis.feature_extraction.models.segmentation.clam import ClamSegmenter

        return ClamSegmenter(freeze=freeze, **build_kwargs)
    else:
        raise ValueError(f"Model type {model_name} not supported")
