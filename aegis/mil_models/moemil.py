from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis.utils.generic_utils import initialize_weights

from .base_mil import BaseMILModel


class MoEMIL(BaseMILModel):
    """
    Instance-Gated Mixture of Experts for Multiple Instance Learning (IG-MoE-MIL).

    This model routes each instance within a bag to one of several "expert"
    sub-networks. A gating network determines the optimal expert for each instance,
    allowing for specialized processing of heterogeneous features within a single bag.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        embed_dim: int = 512,
        num_experts: int = 4,
        dropout_rate: float = 0.25,
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
        self.num_experts = num_experts

        # Gating Network: Decides which expert to use for each instance
        # Takes an instance and outputs a probability distribution over the experts.
        self.gating_network = nn.Sequential(
            nn.Linear(in_dim, num_experts), nn.Softmax(dim=1)
        )

        # Expert Networks: A list of specialized processing pathways
        # Each expert is a small neural network.
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(self.num_experts)
            ]
        )

        # Hierarchical Attention Pooling for aggregation
        self.attention_pool = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.Tanh(),
            nn.Linear(self.embed_dim // 2, 1),
        )

        self.classifier = nn.Linear(self.embed_dim, self.n_classes)
        self.apply(initialize_weights)

    def _forward_impl(self, x: torch.Tensor):
        # x: (batch_size, num_instances, in_dim) - already normalized by base class
        batch_size, num_instances, in_dim = x.shape

        # Flatten all instances to process them in one go
        x_flat = x.view(-1, in_dim)  # (batch_size * num_instances, in_dim)

        # --- Gating and Expert Processing ---
        # Get expert weights for each instance: (batch_size * num_instances, num_experts)
        gating_scores = self.gating_network(x_flat)

        # Calculate outputs for all experts in parallel and combine them
        # expert_outputs will be a list of tensors, each of shape (batch_size * num_instances, embed_dim)
        expert_outputs = [expert(x_flat) for expert in self.experts]

        # Stack outputs for broadcasting: (num_experts, batch_size * num_instances, embed_dim)
        expert_outputs_stacked = torch.stack(expert_outputs)

        # Weight expert outputs by gating scores
        # Unsqueeze gating_scores for broadcasting: (num_experts, batch_size * num_instances, 1)
        gating_scores_expanded = gating_scores.T.unsqueeze(-1)

        # Weighted sum: (batch_size * num_instances, embed_dim)
        weighted_expert_output = torch.sum(
            expert_outputs_stacked * gating_scores_expanded, dim=0
        )

        # Reshape back to bag structure
        # (batch_size, num_instances, embed_dim)
        instance_features = weighted_expert_output.view(
            batch_size, num_instances, self.embed_dim
        )

        # --- Aggregation ---
        # Aggregate instance representations using attention pooling
        attn_weights = self.attention_pool(instance_features)
        attn_weights = F.softmax(attn_weights, dim=1)
        bag_representation = torch.sum(
            instance_features * attn_weights, dim=1
        )  # (batch_size, embed_dim)
        bag_representation = self._fuse_metadata(bag_representation, metadata)

        # --- Classification ---
        logits = self.classifier(bag_representation)
        predictions = torch.topk(logits, 1, dim=1)[1]

        # Return attention weights for interpretability
        instance_attention = attn_weights.squeeze(-1)

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival_curves, predictions, instance_attention, {}
        else:
            probabilities = F.softmax(logits, dim=1)
            return logits, probabilities, predictions, instance_attention, {}
