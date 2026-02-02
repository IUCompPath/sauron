"""
MambaMIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from aegis.mil_models.mamba_ssm.modules.bimamba import BiMamba
    from aegis.mil_models.mamba_ssm.modules.mamba_simple import Mamba
    from aegis.mil_models.mamba_ssm.modules.srmamba import SRMamba

    HAS_MAMBA_SSM = True
except ImportError as e:
    print(f"Warning: mamba_ssm modules could not be imported: {e}")
    HAS_MAMBA_SSM = False
    # Define dummy classes to avoid NameError in __init__ before the check
    BiMamba = object
    Mamba = object
    SRMamba = object

from .base_mil import BaseMILModel


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MambaMIL(BaseMILModel):
    def __init__(
        self,
        in_dim,
        n_classes,
        dropout_rate=0.25,
        dropout=None,  # For backward compatibility
        activation="relu",
        act=None,  # For backward compatibility
        is_survival=False,
        survival=None,  # For backward compatibility
        layer=2,
        rate=10,
        type="SRMamba",
    ):
        # Map backward compatibility parameters
        if dropout is not None:
            dropout_rate = dropout
        if act is not None:
            activation = act
        if survival is not None:
            is_survival = survival

        if not HAS_MAMBA_SSM:
            raise ImportError(
                "MambaMIL cannot be initialized because mamba_ssm modules are missing. "
                "Please install mamba-ssm and causal-conv1d."
            )

        super().__init__(in_dim=in_dim, n_classes=n_classes, is_survival=is_survival)

        self._fc1 = [nn.Linear(in_dim, 512)]
        if activation.lower() == "relu":
            self._fc1 += [nn.ReLU()]
        elif activation.lower() == "gelu":
            self._fc1 += [nn.GELU()]
        if dropout_rate > 0:
            self._fc1 += [nn.Dropout(dropout_rate)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()

        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        elif type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(
                            d_model=512,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.rate = rate
        self.type = type

        self.apply(initialize_weights)

    def _forward_impl(self, x: torch.Tensor):
        # x: (batch_size, num_instances, in_dim) - already normalized by base class
        h = x.float()  # [B, n, in_dim]

        h = self._fc1(h)  # [B, n, 512]

        if self.type == "SRMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h, rate=self.rate)
                h = h + h_
        elif self.type == "Mamba" or self.type == "BiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)
                h = layer[1](h)
                h = h + h_

        h = self.norm(h)
        A = self.attention(h)  # [B, n, 1]
        A = torch.transpose(A, 1, 2)  # [B, 1, n]
        A = F.softmax(A, dim=-1)  # [B, 1, n]
        h = torch.bmm(A, h)  # [B, 1, 512]
        h = h.squeeze(1)  # [B, 512] - squeeze the attention dimension, not batch

        logits = self.classifier(h)  # [B, n_classes]

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            predictions = torch.topk(logits, 1, dim=1)[1]
            # Return attention scores in original shape: (B, n)
            attention_scores = A.squeeze(1)  # [B, n]
            return hazards, survival_curves, predictions, attention_scores, {}
        else:
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.topk(logits, 1, dim=1)[1]
            # Return attention scores in original shape: (B, n)
            attention_scores = A.squeeze(1)  # [B, n]
            return logits, probabilities, predictions, attention_scores, {}

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.layers = self.layers.to(device)

        self.attention = self.attention.to(device)
        self.norm = self.norm.to(device)
        self.classifier = self.classifier.to(device)
