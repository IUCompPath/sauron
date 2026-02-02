from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis.utils.generic_utils import initialize_weights

from .activations import get_activation_fn
from .base_mil import BaseMILModel


class GatedLinear(nn.Module):
    """Gated Linear Unit for feature selection."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.tanh(self.gate(x)) * self.linear(x)


class HGACrossHeadCom(BaseMILModel):
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
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Feature extractor with Gated Linear Unit
        feature_layers = [GatedLinear(in_dim, self.embed_dim)]
        feature_layers.append(get_activation_fn(activation))
        if dropout_rate > 0:
            feature_layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor = nn.Sequential(*feature_layers)

        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Cross-head communication layer
        self.cross_head_comm = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1)

        # Hierarchical attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.Tanh(),
            nn.Linear(self.embed_dim // 2, 1),
        )

        self.classifier = nn.Linear(self.embed_dim, n_classes)
        self.apply(initialize_weights)

    def _forward_impl(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None):
        batch_size, num_instances, _ = x.size()

        instance_features = self.feature_extractor(x)

        q = self.query_proj(instance_features)
        k = self.key_proj(instance_features)
        v = self.value_proj(instance_features)

        q = q.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, num_instances, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Cross-head communication
        q = self.cross_head_comm(q)
        k = self.cross_head_comm(k)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0 if self.training else 0.0
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_instances, self.embed_dim)
        )

        # Hierarchical Attention Pooling
        attn_weights = self.attention_pool(attn_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        bag_representation = torch.sum(attn_output * attn_weights, dim=1)
        bag_representation = self._fuse_metadata(bag_representation, metadata)

        logits = self.classifier(bag_representation)
        predictions = torch.topk(logits, 1, dim=1)[1]

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival_curves, predictions, attn_weights.squeeze(-1), {}
        else:
            probabilities = F.softmax(logits, dim=1)
            return logits, probabilities, predictions, attn_weights.squeeze(-1), {}
