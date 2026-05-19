import os
import sys
import traceback
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from einops import rearrange

from aegis.feature_extraction.utils.io import get_weights_path, has_internet_connection

"""
This file contains an assortment of pretrained slide encoders, all loadable via the encoder_factory() function.
"""


def encoder_factory(
    model_name: str, pretrained: bool = True, freeze: bool = True, **kwargs
) -> torch.nn.Module:
    """
    Build a slide encoder model.

    Args:
        model_name (str): Name of the model to build.
        pretrained (bool): Whether to load pretrained weights.
        freeze (bool): Whether to freeze the weights of the model.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        torch.nn.Module: The slide encoder model.
    """

    if model_name.startswith("mean-"):
        enc = MeanSlideEncoder
        return enc(model_name=model_name)
    elif "threads" in model_name:
        enc = ThreadsSlideEncoder
    elif "titan" in model_name:
        enc = TitanSlideEncoder
    elif "prism" in model_name:
        enc = PRISMSlideEncoder
    elif "chief" in model_name:
        enc = CHIEFSlideEncoder
    elif "gigapath" in model_name:
        enc = GigaPathSlideEncoder
    elif "madeleine" in model_name:
        enc = MadeleineSlideEncoder
    elif (
        "abmil" in model_name
    ):  # This is a generic ABMIL, not a specific pretrained one.
        enc = ABMILSlideEncoder
    else:
        raise ValueError(f"Model type {model_name} not supported")

    return enc(pretrained=pretrained, freeze=freeze, **kwargs)


# Map from slide encoder to required patch encoder
# Used in Processor.py to load the correct patch encoder for a given slide encoder
slide_to_patch_encoder_name = {
    "threads": "conch_v15",
    "titan": "conch_v15",
    "tcga": "conch_v15",
    "prism": "virchow",
    "chief": "ctranspath",
    "gigapath": "gigapath",
    "madeleine": "conch_v1",
    # Mean-pooling models infer their patch encoder from their name.
    # The 'mean-' prefix will be stripped to get the patch encoder name.
}


class BaseSlideEncoder(torch.nn.Module):
    _has_internet = has_internet_connection()

    def __init__(self, freeze: bool = True, **build_kwargs: dict) -> None:
        """
        Parent class for all pretrained slide encoders.
        """
        super().__init__()
        self.enc_name = None
        self.model, self.precision, self.embedding_dim = self._build(**build_kwargs)

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def _get_weights_path(self):
        """
        If self.weights_path is provided (via build_kwargs), use it.
        If not provided, check the model registry.
            If path in model registry is empty, auto-download from huggingface
            else, use the path from the registry.
        """
        weights_path = self.build_kwargs.get(
            "weights_path", None
        )  # Fixed bug: use self.build_kwargs
        if weights_path:
            self.ensure_valid_weights_path(weights_path)
            return weights_path
        else:
            weights_path = get_weights_path("slide", self.enc_name)
            self.ensure_valid_weights_path(weights_path)
            return weights_path

    def ensure_valid_weights_path(self, weights_path):
        if (
            weights_path
            and not os.path.isfile(weights_path)
            and not os.path.isdir(weights_path)
        ):  # CHIEF uses a directory
            raise FileNotFoundError(
                f"Expected checkpoint/directory at '{weights_path}', but it was not found."
            )

    def ensure_has_internet(self, enc_name):
        if not BaseSlideEncoder._has_internet:
            raise FileNotFoundError(
                f"Internet connection does seem not available. Auto checkpoint download is disabled."
                f"To proceed, please manually download: {enc_name},\n"
                f"and place it in the model registry in:\n`aegis/feature_extraction/models/slide_encoders/checkpoints.json`"
            )

    def forward(self, batch: Dict[str, Any], device: str) -> torch.Tensor:
        """
        Can be overwritten if model requires special forward pass.
        `batch` expected to be a dictionary containing 'features', 'coords', 'attributes'.
        """
        z = self.model(batch)
        return z

    @abstractmethod
    def _build(self, **build_kwargs) -> Tuple[torch.nn.Module, torch.dtype, int]:
        """
        Initialization method, must be defined in child class.
        Returns: model, precision, embedding_dim
        """
        pass


class CustomSlideEncoder(BaseSlideEncoder):
    def __init__(
        self,
        enc_name: str,
        model: torch.nn.Module,
        precision: torch.dtype = torch.float32,
        embedding_dim: Optional[int] = None,
        **build_kwargs,  # Capture other kwargs but they won't be used by _build
    ):
        """
        CustomSlideEncoder initialization.

        This class is used when the model and precision are pre-instantiated externally
        and should be injected directly into the encoder wrapper.

        Args:
            enc_name (str):
                A unique name or identifier for the encoder.
            model (torch.nn.Module):
                A PyTorch model instance to use for slide-level inference.
            precision (torch.dtype, optional):
                The precision to use for inference (e.g., torch.float32, torch.float16).
            embedding_dim (int, optional):
                The output embedding dimension. If not provided, will attempt to use
                `model.embedding_dim` if it exists.
        """
        # Call BaseSlideEncoder with freeze=False because freezing is handled externally for custom models
        super().__init__(freeze=False, **build_kwargs)
        self.enc_name = enc_name
        self.model = model
        self.precision = precision
        self.embedding_dim = embedding_dim or getattr(model, "embedding_dim", None)
        if self.embedding_dim is None:
            raise ValueError(
                "For CustomSlideEncoder, embedding_dim must be provided or inferable from model.embedding_dim."
            )

    def _build(self, **build_kwargs):
        # For CustomSlideEncoder, model, precision, embedding_dim are passed directly to __init__
        # and stored. _build just returns these.
        return self.model, self.precision, self.embedding_dim


class ABMILSlideEncoder(BaseSlideEncoder):
    def __init__(self, **build_kwargs):
        """
        ABMIL initialization.
        """
        super().__init__(**build_kwargs)

    def _build(
        self,
        input_feature_dim: int,
        n_heads: int = 8,
        head_dim: int = 256,
        dropout: float = 0.25,
        gated: bool = True,
        pretrained: bool = False,  # This model is not pretrained from a public checkpoint
    ) -> Tuple[torch.nn.ModuleDict, torch.dtype, int]:
        import torch.nn as nn

        from aegis.feature_extraction.models.slide_encoders.zoo.reusable_blocks.ABMIL import (
            ABMIL,
        )

        self.enc_name = "abmil"

        assert (
            pretrained is False
        ), "ABMILSlideEncoder has no corresponding pretrained models. Please load with pretrained=False."

        pre_attention_layers = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim), nn.GELU(), nn.Dropout(0.1)
        )

        image_pooler = ABMIL(
            n_heads=n_heads,
            feature_dim=input_feature_dim,
            head_dim=head_dim,
            dropout=dropout,
            n_branches=1,
            gated=gated,
        )

        post_attention_layers = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim), nn.GELU(), nn.Dropout(0.1)
        )

        model = nn.ModuleDict(
            {
                "pre_attention_layers": pre_attention_layers,
                "image_pooler": image_pooler,
                "post_attention_layers": post_attention_layers,
            }
        )

        precision = torch.float32
        embedding_dim = input_feature_dim
        return model, precision, embedding_dim

    def forward(
        self, batch: Dict[str, Any], device: str, return_raw_attention=False
    ) -> torch.Tensor:
        image_features = self.model["pre_attention_layers"](
            batch["features"].to(device)
        )
        image_features, attn = self.model["image_pooler"](
            image_features
        )  # Features shape: (b n_branches f), where n_branches = 1. Branching is not used in this implementation.
        image_features = rearrange(image_features, "b 1 f -> b f")
        image_features = self.model["post_attention_layers"](
            image_features
        )  # Attention scores shape: (b r h n), where h is number of attention heads
        if return_raw_attention:
            return image_features, attn
        return image_features


class PRISMSlideEncoder(BaseSlideEncoder):
    def __init__(self, **build_kwargs):
        """
        PRISM initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):
        self.enc_name = "prism"

        if sys.version_info < (3, 10):
            raise RuntimeError(
                "PRISM requires Python 3.10 or above. Please update your Python interpreter."
            )

        try:
            import environs  # weird dependencies required by PRISM
            import sacremoses  # type: ignore
            from transformers import AutoConfig, AutoModel  # type: ignore
        except ImportError:
            traceback.print_exc()
            raise ImportError(
                "Please run `pip install environs==11.0.0 transformers==4.42.4 sacremoses==0.1.1` "
                "and ensure Python version is 3.10 or above."
            )

        if pretrained:
            self.ensure_has_internet(self.enc_name)
            model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)
        else:
            model = AutoModel.from_config(AutoConfig.from_pretrained("paige-ai/Prism"))

        # Remove the text decoder as it's not needed for slide encoding
        model.text_decoder = None

        precision = torch.float16
        embedding_dim = 1280
        return model, precision, embedding_dim

    def forward(self, batch: Dict[str, Any], device: str) -> torch.Tensor:
        # input should be of shape (batch_size, tile_seq_len, tile_embed_dim)
        x = batch["features"].to(device)
        # Assuming model.slide_representations takes a batch of features and returns a dict with 'image_embedding'
        z = self.model.slide_representations(x)
        z = z["image_embedding"]
        return z


class CHIEFSlideEncoder(BaseSlideEncoder):
    def __init__(self, **build_kwargs):
        """
        CHIEF initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):
        self.enc_name = "chief"
        weights_dir = (
            self._get_weights_path()
        )  # CHIEF uses a directory as its "weight path"

        # Ensure model can be built.
        try:
            sys.path.append(weights_dir)
            from models.CHIEF import CHIEF  # type: ignore
        except ImportError:
            traceback.print_exc()
            raise ImportError(
                f"\nError: Unable to import the CHIEF repository from '{weights_dir}'.\n\n"
                "To resolve this issue:\n"
                "1. Ensure you have cloned the CHIEF repository to a convenient location:\n"
                "   `git clone https://github.com/hms-dbmi/CHIEF/`\n"
                "2. Set the path to CHIEF repo in `aegis/feature_extraction/models/slide_encoders/checkpoints.json`, e.g., `./CHIEF`.\n"
                "3. Verify that CHIEF dependencies are installed:\n"
                "   `pip install addict`\n\n"
            )

        # Ensure weights can be loaded.
        try:
            current_wd = os.getcwd()  # Get current working directory
            # CHIEF expects to be run from its own directory for loading weights
            os.chdir(weights_dir)
            os.makedirs(os.path.join(weights_dir, "model_weight"), exist_ok=True)

            required_files = {
                "Text_emdding.pth": "https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV",
                "CHIEF_pretraining.pth": "https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV",
            }

            for file_name, download_link in required_files.items():
                file_path = os.path.join(weights_dir, "model_weight", file_name)
                if not os.path.exists(file_path):
                    # In a CI/CD environment or non-interactive, this means manual download/setup is required
                    raise FileNotFoundError(
                        f"\nError: Missing required file '{file_name}' for CHIEF.\n\n"
                        "To resolve this issue:\n"
                        f"1. Download the file from:\n   {download_link}\n"
                        f"2. Copy '{file_name}' to the following directory:\n   {file_path}\n\n"
                        "Ensure the file is correctly placed before retrying."
                    )

            # CHIEF model expects some configuration file in its directory, which is usually handled internally.
            # Assuming the cloned repo has necessary config files.
            print("All necessary files for CHIEF are present. CHIEF setup is complete!")

        except (
            FileNotFoundError
        ) as e:  # Catch FileNotFoundError specifically for missing files
            raise e
        except Exception as e:
            print("\nAn error occurred during CHIEF setup:")
            traceback.print_exc()
            raise e

        # Initialize CHIEF model
        model = CHIEF(size_arg="small", dropout=True, n_classes=2)

        # Load pretrained weights
        if pretrained:
            td = torch.load(
                os.path.join("model_weight", "CHIEF_pretraining.pth"),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(td, strict=True)

        # Return to original working directory
        os.chdir(current_wd)

        precision = torch.float32
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch: Dict[str, Any], device: str) -> torch.Tensor:
        # CHIEF expects (N, D) where N is number of patches, D is feature dim
        # Input batch['features'] is (B, N, D). Squeeze batch dim.
        x = batch["features"].squeeze(0).to(device)
        # CHIEF forward pass expects an additional dummy `text_features` argument, usually a tensor of zeros or ones.
        # Or a batch_size list. Dummy for now for simplicity based on its usage.
        # From CHIEF code: self(x, text_features=torch.tensor([0]))
        # text_features is often a dummy for this path as it's a multimodal model.
        z = self.model(
            x, text_features=torch.tensor([0], device=device)
        )  # Pass dummy text_features
        z = z["WSI_feature"]  # Shape (1,768) - Already squeezed batch dim from outside
        return z


class GigaPathSlideEncoder(BaseSlideEncoder):
    def __init__(self, **build_kwargs):
        """
        GigaPath initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):
        self.enc_name = "gigapath"

        try:
            from gigapath.slide_encoder import create_model  # type: ignore
        except ImportError:
            traceback.print_exc()
            raise ImportError(
                "Please install fairscale and gigapath using `pip install fairscale git+https://github.com/prov-gigapath/prov-gigapath.git`."
            )

        # Make sure flash_attn is correct version
        try:
            import flash_attn

            assert flash_attn.__version__ == "2.5.8"  # type: ignore
        except AssertionError:
            traceback.print_exc()
            raise ImportError(
                "Please install flash_attn version 2.5.8 using `pip install flash_attn==2.5.8`."
            )
        except ImportError:
            traceback.print_exc()
            raise ImportError(
                "flash_attn is not installed. Please install it using `pip install flash_attn` (ensure CUDA compatibility)."
            )

        if pretrained:
            self.ensure_has_internet(self.enc_name)
            model = create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                "gigapath_slide_enc12l768d",
                1536,
                global_pool=True,
            )
        else:
            # When pretrained=False, the first argument to create_model should be an empty string
            model = create_model(
                "", "gigapath_slide_enc12l768d", 1536, global_pool=True
            )

        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch: Dict[str, Any], device: str) -> torch.Tensor:
        # GigaPath requires tile_size to be set on the model
        if "attributes" not in batch or "patch_size_level0" not in batch["attributes"]:
            raise ValueError(
                "Batch must contain 'attributes' with 'patch_size_level0' for GigaPathSlideEncoder."
            )
        self.model.tile_size = batch["attributes"][
            "patch_size_level0"
        ]  # Set dynamically

        # GigaPath forward: features (B, N, D), coords (B, N, 2)
        # Returns (features from different layers). Layer 11 is the last.
        z = self.model(
            batch["features"].to(device),
            batch["coords"].to(device),
            all_layer_embed=True,
        )[11]
        return z


class MadeleineSlideEncoder(BaseSlideEncoder):
    def __init__(self, **build_kwargs):
        """
        Madeleine initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):
        assert (
            pretrained
        ), "MadeleineSlideEncoder has no non-pretrained models. Please load with pretrained=True."

        self.enc_name = "madeleine"
        weights_path = self._get_weights_path()
        embedding_dim = 512

        try:
            from madeleine.models.factory import (
                create_model_from_pretrained,  # type: ignore
            )
        except ImportError:
            traceback.print_exc()
            raise ImportError(
                "Please install Madeleine using `pip install git+https://github.com/mahmoodlab/MADELEINE.git`"
            )

        if not weights_path:
            self.ensure_has_internet(self.enc_name)
            # Madeleine also uses hf_hub_download internally in create_model_from_pretrained if checkpoint_path is a hub ID.
            # Assuming create_model_from_pretrained can handle the hub path directly.
            model, precision = create_model_from_pretrained(
                "hf_hub:MahmoodLab/madeleine"
            )
        else:
            model, precision = create_model_from_pretrained(weights_path)

        return model, precision, embedding_dim

    def forward(self, batch: Dict[str, Any], device: str) -> torch.Tensor:
        # Madeleine expects patch features directly, not a batch dict
        # Features are (B, N, D). Squeeze batch dim.
        x_features = batch["features"].squeeze(0)
        z = self.model.encode_he(x_features, device)
        return z


class ThreadsSlideEncoder(BaseSlideEncoder):
    def __init__(self, **build_kwargs):
        """
        Threads initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):
        self.enc_name = "threads"

        try:
            # Assuming threadsmodel might need to be installed or is available in the environment
            from threadsmodel.inference import (  # type: ignore
                create_model,
                create_model_from_pretrained,
            )
        except ImportError:
            traceback.print_exc()
            raise ImportError(
                "Threads model is coming soon! Thanks for your patience. (Missing threadsmodel dependency)"
            )

        # Threads model is coming soon, so no actual implementation here yet
        # Placeholder for future integration
        print("ThreadsSlideEncoder: Model definition is a placeholder (Coming Soon!)")
        model = torch.nn.Identity()  # Dummy model
        precision = torch.float16  # Default precision
        embedding_dim = 768

        return model, precision, embedding_dim

    def forward(
        self, batch: Dict[str, Any], device: str, return_raw_attention=False
    ) -> torch.Tensor:
        # Placeholder forward for "Coming Soon" model
        print("ThreadsSlideEncoder: Forward pass is a placeholder.")
        # If it's a dummy Identity, it will just return the input features
        # For a slide encoder, it should output a single vector per slide.
        # This needs actual model implementation.
        # For now, return mean-pooled features as a dummy if it is an identity model
        return batch["features"].mean(dim=1).to(device)


class TitanSlideEncoder(BaseSlideEncoder):
    def __init__(self, **build_kwargs):
        """
        Titan initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, pretrained=True):
        self.enc_name = "titan"
        assert (
            pretrained
        ), "TitanSlideEncoder has no non-pretrained models. Please load with pretrained=True."

        try:
            from transformers import AutoModel  # type: ignore

            self.ensure_has_internet(self.enc_name)
            model = AutoModel.from_pretrained(
                "MahmoodLab/TITAN", trust_remote_code=True
            )
        except ImportError:
            traceback.print_exc()
            raise ImportError(
                "Please install transformers and ensure network access for TITAN model."
            )
        except Exception:
            traceback.print_exc()
            raise Exception(
                "Failed to download TITAN model. Make sure you were granted access and correctly registered your token."
            )

        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch: Dict[str, Any], device: str) -> torch.Tensor:
        # Titan expects patch features, coords, and patch_size_level0
        # `batch['features']` is (B, N, D), `batch['coords']` is (B, N, 2)
        # `batch['attributes']['patch_size_level0']` is scalar (int)

        if "attributes" not in batch or "patch_size_level0" not in batch["attributes"]:
            raise ValueError(
                "Batch must contain 'attributes' with 'patch_size_level0' for TitanSlideEncoder."
            )

        z = self.model.encode_slide_from_patch_features(
            batch["features"].to(device),
            batch["coords"].to(device),
            batch["attributes"]["patch_size_level0"],
        )
        return z


class MeanSlideEncoder(BaseSlideEncoder):
    def __init__(self, **build_kwargs):
        """
        Mean pooling initialization.
        """
        super().__init__(**build_kwargs)

    def _build(self, model_name="mean-default"):
        self.enc_name = model_name

        # Determine embedding dimension based on the assumed patch encoder name
        # The 'mean-' prefix will be stripped to get the patch encoder name.
        patch_encoder_name = model_name.replace("mean-", "")

        if patch_encoder_name == "conch_v1":
            embedding_dim = 512
        elif patch_encoder_name == "conch_v15":
            embedding_dim = 768
        elif patch_encoder_name == "uni_v1":
            embedding_dim = 1024
        elif patch_encoder_name == "uni_v2":
            embedding_dim = 1536
        elif patch_encoder_name == "ctranspath":
            embedding_dim = 768
        elif patch_encoder_name == "phikon":
            embedding_dim = 768
        elif patch_encoder_name == "resnet50":
            embedding_dim = 1024
        elif patch_encoder_name == "gigapath":
            embedding_dim = 1536
        elif patch_encoder_name == "virchow":
            embedding_dim = 2560
        elif patch_encoder_name == "virchow2":
            embedding_dim = 2560
        elif patch_encoder_name == "hoptimus0" or patch_encoder_name == "hoptimus1":
            embedding_dim = 1536
        elif patch_encoder_name == "phikon_v2":
            embedding_dim = 1024
        elif patch_encoder_name == "musk":
            embedding_dim = 1024
        elif patch_encoder_name == "hibou_l":
            embedding_dim = 1024
        elif "kaiko" in patch_encoder_name:  # Handle all Kaiko variants
            if "vits8" in patch_encoder_name or "vits16" in patch_encoder_name:
                embedding_dim = 384
            elif "vitb8" in patch_encoder_name or "vitb16" in patch_encoder_name:
                embedding_dim = 768
            elif "vitl14" in patch_encoder_name:
                embedding_dim = 1024
            else:
                embedding_dim = None  # Unknown Kaiko variant
        elif patch_encoder_name == "lunit-vits8":
            embedding_dim = 384
        elif patch_encoder_name == "midnight12k":
            embedding_dim = 3072  # Midnight's default cls+mean output
        else:
            print(
                f"\033[93mWARNING: Could not automatically infer embedding_dim for mean encoder {self.enc_name} from patch encoder {patch_encoder_name}. Setting to None.\033[0m"
            )
            embedding_dim = None

        return (
            None,
            torch.float32,
            embedding_dim,
        )  # No actual model, precision can be float32, embedding_dim inferred.

    def forward(self, batch: Dict[str, Any], device: str) -> torch.Tensor:
        # The mean encoder simply takes the mean of the patch features
        z = batch["features"].to(device).mean(dim=1)  # (B, N, D) -> (B, D)
        return z
