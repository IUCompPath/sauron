from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention, global_max_pool, global_mean_pool

from aegis.utils.generic_utils import initialize_weights

from .activations import get_activation_fn  # Assuming it's in mil_models/
from .base_mil import BaseMILModel

# torch.autograd.set_detect_anomaly(True) # Should be for debugging, not in library code


class WiKG(BaseMILModel):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden_dim: int = 512,
        top_k_neighbors: int = 6,
        agg_type: str = "bi-interaction",  # "gcn", "sage", "bi-interaction"
        pool_type: str = "attn",  # "mean", "max", "attn"
        dropout_rate: float = 0.3,
        activation: str = "leaky_relu",  # Original used LeakyReLU
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

        # Using LeakyReLU as per original if specified, else map via get_activation_fn
        if activation.lower() == "leaky_relu":
            self.activation_fn = nn.LeakyReLU()
        else:
            self.activation_fn = get_activation_fn(activation)

        self.feature_extractor_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            self.activation_fn,  # LeakyReLU or other
        )

        self.W_head = nn.Linear(hidden_dim, hidden_dim)
        self.W_tail = nn.Linear(hidden_dim, hidden_dim)

        self.scale = hidden_dim**-0.5
        self.top_k_neighbors = top_k_neighbors
        self.agg_type = agg_type

        # Gated Knowledge Attention components (Original seems to miss these, but they are usually part of WiKG)
        # Based on typical Gated Attention Units or similar structures
        # For simplicity, I'll follow the provided structure for neighbor aggregation
        # and assume KA (Knowledge Attention) weights are implicitly handled or simplified.
        # If a more complex KA is needed, it would involve more layers.

        # Aggregation layers
        if self.agg_type == "gcn":
            self.agg_linear = nn.Linear(hidden_dim, hidden_dim)
        elif self.agg_type == "sage":
            self.agg_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        elif self.agg_type == "bi-interaction":
            self.agg_linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.agg_linear2 = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise NotImplementedError(f"Aggregation type '{agg_type}' not implemented.")

        self.message_dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.classifier_fc = nn.Linear(hidden_dim, n_classes)

        # Readout (Pooling) layer
        if pool_type == "mean":
            self.readout = global_mean_pool
        elif pool_type == "max":
            self.readout = global_max_pool
        elif pool_type == "attn":
            att_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                self.activation_fn,  # LeakyReLU or other
                nn.Linear(hidden_dim // 2, 1),
            )
            self.readout = GlobalAttention(att_net)
        else:
            raise NotImplementedError(f"Pooling type '{pool_type}' not implemented.")

        self.apply(initialize_weights)

    def _forward_impl(self, x: torch.Tensor):
        # x: (batch_size, num_instances, in_dim) - already normalized by base class
        # Note: Base class handles dict inputs, but this model expects tensor

        batch_size, num_instances, _ = x.shape

        instance_features = self.feature_extractor_fc(x)  # (B, N, C_hidden)

        # Instance graph construction / feature smoothing
        # (Original: (x + x.mean(dim=1, keepdim=True)) * 0.5).
        # This averages each instance feature with the mean feature of the bag.
        instance_features = (
            instance_features + instance_features.mean(dim=1, keepdim=True)
        ) * 0.5

        e_h = self.W_head(instance_features)  # (B, N, C_h) "head" features for query
        e_t = self.W_tail(
            instance_features
        )  # (B, N, C_h) "tail" features for key/value

        # Construct neighborhood via attention
        # attn_logit: (B, N, N) - similarity between each pair of instances in a bag
        attn_logit = torch.bmm(e_h * self.scale, e_t.transpose(1, 2))

        # Select top-k neighbors for each instance
        # topk_weights: (B, N, K_neighbors), topk_indices: (B, N, K_neighbors)
        topk_weights, topk_indices = torch.topk(
            attn_logit, k=self.top_k_neighbors, dim=-1
        )

        # Gather features of top-k neighbors
        # e_t is (B, N, C_h). topk_indices is (B, N, K_neigh).
        # We need to gather along dim 1 (instance dimension) for each batch element.
        # expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, e_t.shape[-1])
        # neighbor_features = torch.gather(e_t.unsqueeze(2).expand(-1, -1, num_instances, -1), 2, expanded_indices)
        # More direct way using advanced indexing:
        batch_idx_for_gather = torch.arange(batch_size, device=x.device).view(-1, 1, 1)
        neighbor_features = e_t[
            batch_idx_for_gather, topk_indices
        ]  # (B, N, K_neigh, C_h)

        # Softmax for attention probabilities over neighbors
        topk_attention_probs = F.softmax(topk_weights, dim=2)  # (B, N, K_neigh)

        # Weighted sum of neighbor features (eh_r in original)
        # This part from original "eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))"
        # is unusual for standard attention. A simple weighted sum is more common.
        # eh_r seems like a mix of neighbor features and self features.
        # For now, let's compute aggregated neighbor representation:
        aggregated_neighbors = torch.einsum(
            "bnk,bnkc->bnc", topk_attention_probs, neighbor_features
        )  # (B, N, C_h)

        # The "gated knowledge attention" part (ka_weight, ka_prob, e_Nh) from original
        # seems to be another layer of attention on these neighbors.
        # For simplicity and robustness, using the aggregated_neighbors directly or a simplified KA.
        # Original KA:
        # e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.top_k_neighbors, -1) # (B, N, K_neigh, C_h)
        # gate_input = torch.tanh(e_h_expand + neighbor_features) # Combine self with neighbors
        # ka_logit = torch.einsum("bnkc,bnkc->bnk", neighbor_features, gate_input) # Dot product
        # ka_attention_probs = F.softmax(ka_logit, dim=2) # (B, N, K_neigh)
        # e_Nh_refined_neighbors = torch.einsum('bnk,bnkc->bnc', ka_attention_probs, neighbor_features) # (B,N,C_h)
        # This e_Nh_refined_neighbors is the `e_Nh` from original. Let's use this refined version.
        e_h_expanded = e_h.unsqueeze(2).expand_as(neighbor_features)
        gate_val = torch.tanh(e_h_expanded + neighbor_features)  # (B,N,K,C)
        ka_weights = torch.sum(neighbor_features * gate_val, dim=-1)  # (B,N,K)
        ka_probs = F.softmax(ka_weights, dim=-1)  # (B,N,K)
        e_Nh = torch.sum(ka_probs.unsqueeze(-1) * neighbor_features, dim=2)  # (B,N,C)

        # Node feature aggregation (GCN, SAGE, Bi-interaction style)
        if self.agg_type == "gcn":
            aggregated_embedding = self.activation_fn(self.agg_linear(e_h + e_Nh))
        elif self.agg_type == "sage":
            concat_embedding = torch.cat([e_h, e_Nh], dim=2)
            aggregated_embedding = self.activation_fn(self.agg_linear(concat_embedding))
        elif self.agg_type == "bi-interaction":
            sum_embedding = self.activation_fn(self.agg_linear1(e_h + e_Nh))
            bi_embedding = self.activation_fn(self.agg_linear2(e_h * e_Nh))
            aggregated_embedding = sum_embedding + bi_embedding
        else:  # Should not happen due to init check
            aggregated_embedding = e_h

        h_messages = self.message_dropout(aggregated_embedding)  # (B, N, C_h)

        # Readout to get bag-level representation
        # Reshape for torch_geometric global pooling: (TotalNodes, Features)
        h_reshaped = h_messages.contiguous().view(-1, h_messages.size(-1))
        # Create batch vector for pooling: [0,0..0, 1,1..1, ..., B-1..B-1]
        batch_vector = torch.arange(batch_size, device=x.device).repeat_interleave(
            num_instances
        )

        bag_representation = self.readout(h_reshaped, batch=batch_vector)  # (B, C_h)
        bag_representation = self.output_norm(bag_representation)
        bag_representation = self._fuse_metadata(bag_representation, metadata)
        logits = self.classifier_fc(bag_representation)  # (B, n_classes)

        # Predictions
        predictions = torch.topk(logits, 1, dim=1)[1]

        # A_raw could be topk_attention_probs or ka_probs. Let's use KA.
        # Need to decide which attention score to return, ka_probs is instance-neighbor level.
        # For bag level, maybe average instance attention? For now, None.
        attention_scores_raw = None

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival_curves, predictions, attention_scores_raw, {}
        else:
            probabilities = F.softmax(logits, dim=1)
            return logits, probabilities, predictions, attention_scores_raw, {}
