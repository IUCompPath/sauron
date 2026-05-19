from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis.utils.generic_utils import initialize_weights  # Assuming this exists

from .activations import get_activation_fn
from .base_mil import BaseMILModel


class _BaseAttentionMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        attention_hidden_dim: int,
        num_attention_outputs: int,  # Typically K=1 for MIL aggregation
        n_classes: int,
        dropout_rate: float = 0.25,
        activation: str = "relu",
        is_survival: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.num_attention_outputs = num_attention_outputs  # K
        self.is_survival = is_survival
        self.n_classes = n_classes

        feature_extractor_layers = [nn.Linear(in_dim, self.embed_dim)]
        feature_extractor_layers.append(get_activation_fn(activation))
        if dropout_rate > 0:
            feature_extractor_layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        self.classifier_layer = nn.Linear(
            self.embed_dim * self.num_attention_outputs, n_classes
        )
        # self.apply(initialize_weights) # Apply if this is a common practice for all sub-modules

    def _get_outputs(self, logits, attention_scores=None):
        # Predictions (highest logit index)
        predictions = torch.topk(logits, 1, dim=1)[1]

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival_curves, predictions, attention_scores, {}
        else:
            probabilities = F.softmax(logits, dim=1)
            return logits, probabilities, predictions, attention_scores, {}


class DAttention(
    BaseMILModel
):  # Original DAttention renamed for clarity if _BaseAttentionMIL is used
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
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
            metadata_fusion_dim=metadata_fusion_dim or 512,
            **kwargs,
        )
        self.embed_dim = 512  # L
        self.attention_hidden_dim = 512  # D
        self.num_attention_outputs = 1  # K

        feature_layers = [nn.Linear(in_dim, self.embed_dim)]
        feature_layers.append(get_activation_fn(activation))
        if dropout_rate > 0:
            feature_layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*feature_layers)

        self.attention_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.attention_hidden_dim, self.num_attention_outputs),
        )
        self.classifier = nn.Linear(
            self.embed_dim * self.num_attention_outputs, n_classes
        )
        self.apply(initialize_weights)

    def _forward_impl(
        self,
        input_tensor: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ):
        # input_tensor: (batch_size, num_instances, in_dim) - already normalized by base class
        batch_size = input_tensor.shape[0]

        instance_features = self.feature_extractor(
            input_tensor
        )  # (batch_size, num_instances, embed_dim)

        attention_logits = self.attention_net(
            instance_features
        )  # (batch_size, num_instances, K)
        # Ensure 3D tensor: if K=1, might be squeezed to 2D
        if attention_logits.dim() == 2:
            attention_logits = attention_logits.unsqueeze(
                -1
            )  # (batch_size, num_instances, 1)
        attention_logits = torch.transpose(
            attention_logits, 2, 1
        )  # (batch_size, K, num_instances)

        # Create mask for padded instances (where all features are zero)
        # Check if all features in an instance are zero (padded)
        # input_tensor is now guaranteed to be 3D after handling above
        instance_mask = (
            input_tensor.abs().sum(dim=-1) > 1e-6
        )  # (batch_size, num_instances)
        # Expand mask to match attention_logits shape: (batch_size, K, num_instances)
        instance_mask = instance_mask.unsqueeze(1).expand_as(attention_logits)

        # Apply mask: set attention logits to very negative value for padded instances
        attention_logits = attention_logits.masked_fill(~instance_mask, float("-inf"))

        attention_scores = F.softmax(attention_logits, dim=2)  # Softmax over instances

        # M = KxL equivalent for batch: (batch_size, K, embed_dim)
        # instance_features should be 3D: (batch_size, num_instances, embed_dim)
        aggregated_features = torch.bmm(attention_scores, instance_features)

        # If K=1, aggregated_features is (batch_size, 1, embed_dim)
        # Flatten for classifier: (batch_size, K * embed_dim)
        aggregated_features_flat = aggregated_features.view(batch_size, -1)
        aggregated_features_flat = self._fuse_metadata(
            aggregated_features_flat, metadata
        )

        logits = self.classifier(aggregated_features_flat)  # (batch_size, n_classes)

        # Predictions (highest logit index)
        predictions = torch.topk(logits, 1, dim=1)[1]

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return (
                hazards,
                survival_curves,
                predictions,
                attention_logits.transpose(2, 1),
                {},
            )  # Return raw attention before softmax
        else:
            probabilities = F.softmax(logits, dim=1)
            return (
                logits,
                probabilities,
                predictions,
                attention_logits.transpose(2, 1),
                {},
            )


class GatedAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        dropout_rate: float = 0.25,
        activation: str = "relu",
        is_survival: bool = False,
    ):
        super().__init__()
        self.embed_dim = 512  # L
        self.attention_hidden_dim = 128  # D
        self.num_attention_outputs = 1  # K

        self.is_survival = is_survival
        self.n_classes = n_classes

        feature_layers = [nn.Linear(in_dim, self.embed_dim)]
        feature_layers.append(get_activation_fn(activation))
        if dropout_rate > 0:
            feature_layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*feature_layers)

        self.attention_V = nn.Sequential(
            nn.Linear(self.embed_dim, self.attention_hidden_dim), nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.embed_dim, self.attention_hidden_dim), nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(
            self.attention_hidden_dim, self.num_attention_outputs
        )

        self.classifier = nn.Linear(
            self.embed_dim * self.num_attention_outputs, n_classes
        )
        self.apply(initialize_weights)

    def forward(self, input_tensor: torch.Tensor):
        # input_tensor: (batch_size, num_instances, in_dim) or (num_instances, in_dim)

        # Handle both 2D and 3D input tensors
        if input_tensor.dim() == 2:
            # If 2D, add batch dimension: (num_instances, in_dim) -> (1, num_instances, in_dim)
            input_tensor = input_tensor.unsqueeze(0)
            batch_size = 1
        elif input_tensor.dim() == 3:
            batch_size = input_tensor.shape[0]
        else:
            raise ValueError(
                f"Expected input_tensor to be 2D or 3D, got {input_tensor.dim()}D"
            )

        instance_features = self.feature_extractor(
            input_tensor
        )  # (batch_size, num_instances, embed_dim)

        attention_values = self.attention_V(
            instance_features
        )  # (batch_size, num_instances, D)
        attention_units = self.attention_U(
            instance_features
        )  # (batch_size, num_instances, D)

        # Element-wise multiplication, then pass through weights layer
        unnormalized_attention_scores = self.attention_weights(
            attention_values * attention_units
        )  # (batch_size, num_instances, K)
        # Ensure 3D tensor: if K=1, might be squeezed to 2D
        if unnormalized_attention_scores.dim() == 2:
            unnormalized_attention_scores = unnormalized_attention_scores.unsqueeze(
                -1
            )  # (batch_size, num_instances, 1)
        unnormalized_attention_scores = torch.transpose(
            unnormalized_attention_scores, 2, 1
        )  # (batch_size, K, num_instances)

        # Create mask for padded instances (where all features are zero)
        # input_tensor is now guaranteed to be 3D after handling above
        instance_mask = (
            input_tensor.abs().sum(dim=-1) > 1e-6
        )  # (batch_size, num_instances)
        # Expand mask to match attention_logits shape: (batch_size, K, num_instances)
        instance_mask = instance_mask.unsqueeze(1).expand_as(
            unnormalized_attention_scores
        )

        # Apply mask: set attention logits to very negative value for padded instances
        unnormalized_attention_scores = unnormalized_attention_scores.masked_fill(
            ~instance_mask, float("-inf")
        )

        normalized_attention_scores = F.softmax(
            unnormalized_attention_scores, dim=2
        )  # Softmax over instances

        # instance_features should be 3D: (batch_size, num_instances, embed_dim)
        aggregated_features = torch.bmm(
            normalized_attention_scores, instance_features
        )  # (batch_size, K, embed_dim)
        flattened_aggregated_features = aggregated_features.view(
            batch_size, -1
        )  # (batch_size, K * embed_dim)

        logits = self.classifier(
            flattened_aggregated_features
        )  # (batch_size, n_classes)

        # Predictions (highest logit index)
        predictions = torch.topk(logits, 1, dim=1)[1]

        if self.is_survival:
            hazard_rates = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazard_rates, dim=1)
            # A_raw for GatedAttention is less direct, similar to DAttention, returning unnormalized_attention_scores before softmax
            return (
                hazard_rates,
                survival_curves,
                predictions,
                unnormalized_attention_scores.transpose(2, 1),
                {},
            )
        else:
            probabilities = F.softmax(logits, dim=1)
            return (
                logits,
                probabilities,
                predictions,
                unnormalized_attention_scores.transpose(2, 1),
                {},
            )
