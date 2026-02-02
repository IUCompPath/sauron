from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis.utils.generic_utils import initialize_weights

from .activations import get_activation_fn
from .base_mil import BaseMILModel


class DifferentiableAttentionMIL(BaseMILModel):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        embed_dim: int = 512,
        num_heads: int = 8,
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
            metadata_fusion_dim=metadata_fusion_dim or embed_dim,
            **kwargs,
        )
        self.embed_dim = embed_dim  # L
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        feature_layers = [nn.Linear(in_dim, self.embed_dim)]
        feature_layers.append(get_activation_fn(activation))
        if dropout_rate > 0:
            feature_layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*feature_layers)

        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Output projection from attention, not used in original scaled_dot_product version for bag_repr
        # self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.classifier = nn.Linear(self.embed_dim, n_classes)
        self.apply(initialize_weights)

    def _forward_impl(self, x: torch.Tensor):
        # x: (batch_size, num_instances, in_dim) - already normalized by base class
        batch_size, num_instances, _ = x.size()

        instance_features = self.feature_extractor(
            x
        )  # (batch_size, num_instances, embed_dim)

        q = self.query_proj(instance_features)  # (batch_size, num_instances, embed_dim)
        k = self.key_proj(instance_features)  # (batch_size, num_instances, embed_dim)
        v = self.value_proj(instance_features)  # (batch_size, num_instances, embed_dim)

        # Reshape for multi-head attention for F.scaled_dot_product_attention
        # (batch_size, num_heads, num_instances, head_dim)
        q = q.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # scaled_dot_product_attention expects (..., S, E) for query, (..., L, E) for key/value
        # Here, S = num_instances, L = num_instances, E = head_dim
        # Input q, k, v: (batch_size, num_heads, num_instances, head_dim)
        # F.scaled_dot_product_attention will operate on last 3 dims if N>3
        # Or reshape to (batch_size * num_heads, num_instances, head_dim)
        # q_reshaped = q.contiguous().view(batch_size * self.num_heads, num_instances, self.head_dim)
        # k_reshaped = k.contiguous().view(batch_size * self.num_heads, num_instances, self.head_dim)
        # v_reshaped = v.contiguous().view(batch_size * self.num_heads, num_instances, self.head_dim)
        # attn_output = F.scaled_dot_product_attention(q_reshaped, k_reshaped, v_reshaped)
        # attn_output = attn_output.view(batch_size, self.num_heads, num_instances, self.head_dim)

        # Direct passing if Pytorch version supports it
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        # attn_output shape: (batch_size, num_heads, num_instances, head_dim)

        # Reshape back and combine heads
        # (batch_size, num_instances, num_heads, head_dim) -> (batch_size, num_instances, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_instances, self.embed_dim)
        )

        # Aggregate over instances (mean pooling of instance representations after attention)
        bag_representation = attn_output.mean(dim=1)  # (batch_size, embed_dim)
        bag_representation = self._fuse_metadata(bag_representation, metadata)

        logits = self.classifier(bag_representation)  # (batch_size, n_classes)

        # Predictions (highest logit index)
        predictions = torch.topk(logits, 1, dim=1)[1]

        # A_raw (attention weights) is not directly returned by F.scaled_dot_product_attention
        # To get it, one would compute (Q @ K.transpose(-2, -1) / sqrt(dim_k)).softmax(dim=-1)
        attention_scores_raw = None  # Placeholder

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival_curves, predictions, attention_scores_raw, {}
        else:
            probabilities = F.softmax(logits, dim=1)
            return logits, probabilities, predictions, attention_scores_raw, {}
