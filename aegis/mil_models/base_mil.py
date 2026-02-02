"""
Base wrapper class for all MIL models.
Provides standardized input/output handling and batch size support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class BaseMILModel(nn.Module):
    """
    Base class for all MIL models that standardizes:
    - Input handling (supports both 2D and 3D inputs)
    - Output format (standardized tuple)
    - Batch size support (automatic handling)

    All MIL models should inherit from this class and implement:
    - `_forward_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]`

    The `_forward_impl` method should return a tuple with:
    - For classification: (logits, probabilities, predictions, attention_scores, metadata)
    - For survival: (hazards, survival_curves, predictions, attention_scores, metadata)

    The base class will handle:
    - Input normalization (2D -> 3D)
    - Output standardization
    - Batch size handling
    """

    def __init__(
        self, in_dim: int, n_classes: int, is_survival: bool = False, **kwargs
    ):
        """
        Initialize the base MIL model.

        Args:
            in_dim: Input feature dimension
            n_classes: Number of output classes
            is_survival: Whether this is a survival analysis task
            **kwargs: Additional arguments (passed to subclasses)
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.is_survival = is_survival

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
                f"Expected input tensor to be 2D or 3D, got {x.ndim}D. "
                f"Shape: {x.shape}"
            )
        return x, batch_size

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
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass with automatic batch size handling.

        Args:
            x: Input tensor of shape (num_instances, in_dim) or (batch_size, num_instances, in_dim)

        Returns:
            Standardized tuple:
            - For classification: (logits, probabilities, predictions, attention_scores, {})
            - For survival: (hazards, survival_curves, predictions, attention_scores, {})
        """
        # Normalize input to 3D
        x_normalized, batch_size = self._normalize_input(x)

        # Call the implementation
        outputs = self._forward_impl(x_normalized)

        # Standardize outputs
        standardized_outputs = self._standardize_output(outputs, batch_size)

        # Note: We always return outputs with batch dimension to maintain consistency
        # with training code expectations. The batch dimension is preserved even for
        # single samples to ensure compatibility with DataLoader outputs.
        return standardized_outputs

    def _forward_impl(self, x: torch.Tensor) -> Tuple[Any, ...]:
        """
        Implementation of the forward pass.
        Subclasses must implement this method.

        Args:
            x: Normalized 3D input tensor (batch_size, num_instances, in_dim)

        Returns:
            Tuple of outputs (format depends on is_survival flag)
        """
        raise NotImplementedError("Subclasses must implement _forward_impl method")
