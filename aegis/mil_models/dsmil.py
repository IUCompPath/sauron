from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis.utils.generic_utils import initialize_weights

from .base_mil import BaseMILModel


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(
        self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False
    ):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh()
            )
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v), nn.Linear(input_size, input_size), nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(
            c, 0, descending=True
        )  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(
            feats, dim=0, index=m_indices[0, :]
        )  # select critical instances, m_feats in shape C x K
        q_max = self.q(
            m_feats
        )  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(
            Q, q_max.transpose(0, 1)
        )  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(
            A
            / torch.sqrt(
                torch.tensor(Q.shape[1], dtype=torch.float32, device=Q.device)
            ),
            0,
        )  # normalize attention scores, A in shape N x C,
        B = torch.mm(
            A.transpose(0, 1), V
        )  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class DSMIL(BaseMILModel):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
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
            metadata_fusion_dim=metadata_fusion_dim or 512,
            **kwargs,
        )

        # Create feature extractor (FCLayer equivalent)
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

        # Create IClassifier
        self.i_classifier = IClassifier(self.feature_extractor, 512, n_classes)

        # Create BClassifier
        self.b_classifier = BClassifier(512, n_classes, dropout_v=dropout_rate)

        self.apply(initialize_weights)

    def _forward_impl(self, x: torch.Tensor):
        # x: (batch_size, num_instances, in_dim) - already normalized by base class
        batch_size, num_instances, _ = x.shape

        # Flatten for instance classifier: (batch_size * num_instances, in_dim)
        x_flat = x.view(-1, x.shape[-1])

        # Get instance features and class scores
        feats, classes = self.i_classifier(
            x_flat
        )  # feats: (B*N, 512), classes: (B*N, n_classes)

        # Reshape back to batch structure for bag classifier
        # BClassifier expects (N, K) and (N, C) where N is number of instances
        # We need to process each bag separately or adapt the classifier
        # For now, let's process each bag in the batch
        bag_logits_list = []
        attention_list = []

        for b in range(batch_size):
            start_idx = b * num_instances
            end_idx = (b + 1) * num_instances
            bag_feats = feats[start_idx:end_idx]  # (N, 512)
            bag_classes = classes[start_idx:end_idx]  # (N, n_classes)

            # BClassifier expects (N, K) and (N, C)
            prediction_bag, A, B = self.b_classifier(bag_feats, bag_classes)
            # prediction_bag: (1, n_classes), A: (N, n_classes), B: (n_classes, 512)

            bag_logits_list.append(prediction_bag.squeeze(0))  # (n_classes,)
            attention_list.append(A)  # (N, n_classes)

        # Stack results
        bag_logits = torch.stack(bag_logits_list)  # (batch_size, n_classes)
        bag_logits = self._fuse_metadata_logits(bag_logits, metadata)
        # For attention, we'll return the first bag's attention or average
        attention_scores = attention_list[0] if len(attention_list) > 0 else None

        # Predictions
        predictions = torch.topk(bag_logits, 1, dim=1)[1]

        if self.is_survival:
            hazards = torch.sigmoid(bag_logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival_curves, predictions, attention_scores, {}
        else:
            probabilities = F.softmax(bag_logits, dim=1)
            return bag_logits, probabilities, predictions, attention_scores, {}
