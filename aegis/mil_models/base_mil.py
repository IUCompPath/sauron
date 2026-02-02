"""
Base wrapper class for all MIL models.
Provides standardized input/output handling and batch size support.
Supports optional multi-modal metadata (e.g. OncoTreeSiteCode) via a sub-network
fused in the later parts of the model (after bag representation).
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseMILModel(nn.Module):
    """
    Base class for all MIL models that standardizes:
    - Input handling (supports both 2D and 3D inputs)
    - Output format (standardized tuple)
    - Batch size support (automatic handling)
    - Multi-modal metadata fusion

    Metadata Fusion Strategy:
    -------------------------
    This class implements a robust "Concatenation + Projection" strategy for fusing
    slide-level metadata (e.g., age, site code) with the bag representation.
    
    Old behavior (deprecated): bag_rep + encoded_metadata
    New behavior (v2): fusion_net(concat([bag_rep, encoded_metadata]))

    Why this is better:
    1. Modality Integrity: Concatenation prevents metadata from "washing out" or corrupting
       subtle image features, which can happen with simple addition.
    2. Maskability: The system is designed to be "maskable". If metadata is missing (None),
       a zero-vector is used. The subsequent LayerNorm in `fusion_net` ensures the
       distribution remains stable, allowing the model to perform well even with missing data.
    3. Learnable Mixing: The projection layer allows the model to learn exactly how much
       weight to assign to the metadata features versus the image features.

    All MIL models should inherit from this class and implement:
    - `_forward_impl(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]`

    The `_forward_impl` method should return a tuple with:
    - For classification: (logits, probabilities, predictions, attention_scores, metadata)
    - For survival: (hazards, survival_curves, predictions, attention_scores, metadata)

    When metadata is provided, subclasses should call `_fuse_metadata(bag_rep, metadata)`
    before the classifier (for models with a bag representation), or
    `_fuse_metadata_logits(bag_logits, metadata)` for models that output bag-level logits directly.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        is_survival: bool = False,
        metadata_dim: int = 0,
        metadata_fusion_dim: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the base MIL model.

        Args:
            in_dim: Input feature dimension
            n_classes: Number of output classes
            is_survival: Whether this is a survival analysis task
            metadata_dim: If > 0, build a metadata sub-network and fuse in later parts
            metadata_fusion_dim: Output dim of metadata encoder (must match bag_rep dim for
                _fuse_metadata; also used for _fuse_metadata_logits). Defaults to in_dim.
            **kwargs: Additional arguments (passed to subclasses)
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.is_survival = is_survival
        self.metadata_dim = metadata_dim
        self.metadata_fusion_dim = metadata_fusion_dim or in_dim

        if metadata_dim > 0:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_dim, self.metadata_fusion_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.metadata_fusion_dim, self.metadata_fusion_dim),
            )
            # Fusion net to combine bag representation and encoded metadata
            # Projects back to in_dim to maintain compatibility with existing classifiers
            self.fusion_net = nn.Sequential(
                nn.Linear(in_dim + self.metadata_fusion_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(in_dim),
            )
            self.metadata_to_logits = nn.Linear(
                self.metadata_fusion_dim, n_classes
            )  # for models that fuse at logits (e.g. Mean/Max MIL)
        else:
            self.metadata_encoder = None
            self.fusion_net = None
            self.metadata_to_logits = None

    def _normalize_input(self, x) -> Tuple[torch.Tensor, int]:
        """
        Normalize input to always be 3D (batch_size, num_instances, in_dim).

        Args:
            x: Input tensor of shape (num_instances, in_dim) or (batch_size, num_instances, in_dim)
               or dict with 'feature' key

        Returns:
            Tuple of (normalized_tensor, batch_size)
        """
        # Handle dict inputs (for models like WiKG)
        if isinstance(x, dict):
            x = x.get("feature")
            if x is None:
                raise ValueError("Input dictionary must contain 'feature' key.")

        if x.ndim == 2:
            # Single bag: (num_instances, in_dim) -> (1, num_instances, in_dim)
            x = x.unsqueeze(0)
            batch_size = 1
        elif x.ndim == 3:
            # Batch of bags: (batch_size, num_instances, in_dim)
            batch_size = x.shape[0]
        else:
            raise ValueError(
                f"Expected input tensor to be 2D or 3D, got {x.ndim}D. Shape: {x.shape}"
            )
        return x, batch_size

    def _fuse_metadata(
        self,
        bag_rep: torch.Tensor,
        metadata: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse encoded metadata with bag representation (before classifier).
        Uses concatenation followed by projection to preserve information from both modalities.

        Args:
            bag_rep: (batch_size, in_dim)
            metadata: (batch_size, metadata_dim) or None

        Returns:
            bag_rep (normalized) if no metadata, else fusion_net(concat([bag_rep, encoded_metadata]))
        """
        if self.metadata_encoder is None:
            return bag_rep

        if metadata is None:
            # Masked/Missing metadata: use zeros to maintain consistent distribution
            metadata_encoded = torch.zeros(
                bag_rep.shape[0], self.metadata_fusion_dim, device=bag_rep.device
            )
        else:
            metadata_encoded = self.metadata_encoder(metadata)

        fused = torch.cat([bag_rep, metadata_encoded], dim=-1)
        return self.fusion_net(fused)

    def _fuse_metadata_logits(
        self,
        bag_logits: torch.Tensor,
        metadata: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse encoded metadata at logits level (for models without a single bag_rep).
        Uses addition at the logits level as a late-stage bias.

        Args:
            bag_logits: (batch_size, n_classes)
            metadata: (batch_size, metadata_dim) or None

        Returns:
            bag_logits unchanged if no metadata, else bag_logits + metadata_to_logits(metadata_encoder(metadata))
        """
        if metadata is None or self.metadata_encoder is None:
            return bag_logits
        return bag_logits + self.metadata_to_logits(self.metadata_encoder(metadata))

    def _standardize_output(
        self, outputs: Tuple[Any, ...], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Standardize model outputs to a consistent format.

        Args:
            outputs: Raw outputs from _forward_impl
            batch_size: Batch size (used for validation)

        Returns:
            Standardized tuple: (primary_output, secondary_output, predictions, attention_scores, metadata)
            - For classification: (logits, probabilities, predictions, attention_scores, {})
            - For survival: (hazards, survival_curves, predictions, attention_scores, {})
        """
        # Handle different output formats
        if len(outputs) == 2:
            # Some models return only (hazards, survival_curves) or (logits, probs)
            # We need to infer the format
            output1, output2 = outputs

            # Check if it's survival format (hazards, survival_curves)
            if self.is_survival:
                hazards = output1
                survival_curves = output2
                # Ensure proper shapes
                if hazards.ndim == 1:
                    hazards = hazards.unsqueeze(0)
                if survival_curves.ndim == 1:
                    survival_curves = survival_curves.unsqueeze(0)

                predictions = (
                    torch.topk(hazards, 1, dim=-1)[1]
                    if hazards.ndim > 1
                    else torch.argmax(hazards).unsqueeze(0)
                )
                return hazards, survival_curves, predictions, None, {}
            else:
                # Classification format
                logits = output1
                probabilities = output2
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                if probabilities.ndim == 1:
                    probabilities = probabilities.unsqueeze(0)

                predictions = (
                    torch.topk(logits, 1, dim=-1)[1]
                    if logits.ndim > 1
                    else torch.argmax(logits).unsqueeze(0)
                )
                return logits, probabilities, predictions, None, {}

        elif len(outputs) == 3:
            # Some models return (logits, probs, predictions) or similar
            output1, output2, output3 = outputs

            if self.is_survival:
                hazards = output1
                survival_curves = output2
                predictions = (
                    output3
                    if isinstance(output3, torch.Tensor)
                    else torch.topk(output1, 1, dim=-1)[1]
                )
                return hazards, survival_curves, predictions, None, {}
            else:
                logits = output1
                probabilities = output2
                predictions = (
                    output3
                    if isinstance(output3, torch.Tensor)
                    else torch.topk(output1, 1, dim=-1)[1]
                )
                return logits, probabilities, predictions, None, {}

        elif len(outputs) == 4:
            # Standard format: (primary, secondary, predictions, attention)
            output1, output2, output3, output4 = outputs

            if self.is_survival:
                hazards = output1
                survival_curves = output2
                predictions = (
                    output3
                    if isinstance(output3, torch.Tensor)
                    else torch.topk(output1, 1, dim=-1)[1]
                )
                attention = output4
                return hazards, survival_curves, predictions, attention, {}
            else:
                logits = output1
                probabilities = output2
                predictions = (
                    output3
                    if isinstance(output3, torch.Tensor)
                    else torch.topk(output1, 1, dim=-1)[1]
                )
                attention = output4
                return logits, probabilities, predictions, attention, {}

        elif len(outputs) == 5:
            # Full format: (primary, secondary, predictions, attention, metadata)
            return outputs

        else:
            raise ValueError(
                f"Unexpected number of outputs: {len(outputs)}. "
                f"Expected 2-5 outputs. Got: {outputs}"
            )

    def forward(
        self,
        x: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass with automatic batch size handling.

        Args:
            x: Input tensor of shape (num_instances, in_dim) or (batch_size, num_instances, in_dim)
            metadata: Optional (batch_size, metadata_dim) for multi-modal fusion in later parts

        Returns:
            Standardized tuple:
            - For classification: (logits, probabilities, predictions, attention_scores, {})
            - For survival: (hazards, survival_curves, predictions, attention_scores, {})
        """
        # Normalize input to 3D
        x_normalized, batch_size = self._normalize_input(x)

        # Call the implementation (subclasses may fuse metadata before classifier)
        outputs = self._forward_impl(x_normalized, metadata=metadata)

        # Standardize outputs
        standardized_outputs = self._standardize_output(outputs, batch_size)

        # Note: We always return outputs with batch dimension to maintain consistency
        # with training code expectations. The batch dimension is preserved even for
        # single samples to ensure compatibility with DataLoader outputs.
        return standardized_outputs

    def _forward_impl(
        self,
        x: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> Tuple[Any, ...]:
        """
        Implementation of the forward pass.
        Subclasses must implement this method.
        When metadata is not None, fuse it before the classifier using
        _fuse_metadata(bag_rep, metadata) or _fuse_metadata_logits(bag_logits, metadata).

        Args:
            x: Normalized 3D input tensor (batch_size, num_instances, in_dim)
            metadata: Optional (batch_size, metadata_dim) for multi-modal fusion

        Returns:
            Tuple of outputs (format depends on is_survival flag)
        """
        raise NotImplementedError("Subclasses must implement _forward_impl method")
