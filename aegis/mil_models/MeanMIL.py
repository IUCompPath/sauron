from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis.utils.generic_utils import initialize_weights

from .activations import get_activation_fn
from .base_mil import BaseMILModel


class MeanMIL(BaseMILModel):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden_dim: int = 512,  # Renamed from fixed 512
        dropout_rate: float = 0.25,
        activation: str = "relu",
        is_survival: bool = False,
        metadata_dim: int = 0,
        metadata_fusion_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            in_dim=in_dim,
            n_classes=n_classes,
            is_survival=is_survival,
            metadata_dim=metadata_dim,
            metadata_fusion_dim=metadata_fusion_dim or hidden_dim,
            **kwargs,
        )

        head_layers = [nn.Linear(in_dim, hidden_dim)]
        head_layers.append(get_activation_fn(activation))
        if dropout_rate > 0:
            head_layers.append(nn.Dropout(dropout_rate))
        # Final layer to n_classes (instance scores/logits)
        head_layers.append(nn.Linear(hidden_dim, n_classes))

        self.instance_scorer = nn.Sequential(*head_layers)
        self.apply(initialize_weights)

    def _forward_impl(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None):
        # x: (batch_size, num_instances, in_dim) - already normalized by base class

        # Get instance-level scores/logits
        instance_logits = self.instance_scorer(
            x
        )  # (batch_size, num_instances, n_classes)

        # Mean pooling over instances for each class logit
        # Result: (batch_size, n_classes)
        bag_logits = torch.mean(instance_logits, dim=1)
        bag_logits = self._fuse_metadata_logits(bag_logits, metadata)

        # Predictions (highest logit index)
        predictions = torch.topk(bag_logits, 1, dim=1)[1]

        if self.is_survival:
            hazards = torch.sigmoid(bag_logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival_curves, predictions, None, {}
        else:
            probabilities = F.softmax(bag_logits, dim=1)
            return bag_logits, probabilities, predictions, None, {}
