import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from aegis.utils.generic_utils import initialize_weights

from .activations import get_activation_fn
from .base_mil import BaseMILModel

# S4DKernel and S4D components (assumed to be correct and kept as is, minor style adjustments)
# _c2r and _r2c are utility functions for complex numbers, often defined locally or imported
_c2r = torch.view_as_real
_r2c = torch.view_as_complex


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability has to be in [0, 1), but got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed
        # self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p) # Not used

    def forward(self, X):
        if self.training and self.p > 0:
            if not self.transposed:
                X = X.transpose(
                    -1, -2
                )  # More robust than rearrange for (B H L) -> (B L H)

            # Determine mask shape
            # For (B, H, L) if tie=True, mask is (B, H, 1) to broadcast over L
            # If tie=False, mask is (B, H, L)
            mask_shape = X.shape[:-1] + (1,) if self.tie else X.shape

            # Original had X.shape[:2], which assumed X was (B, D, ...).
            # If X is (B, H, L), then X.shape[:2] is (B,H) for tied mask over L.
            # mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape

            mask = (
                torch.rand(*mask_shape, device=X.device) > self.p
            )  # Inverted logic for mask: 1 means keep
            X = X * mask * (1.0 / (1.0 - self.p))  # Scale by 1/(1-p)

            if not self.transposed:
                X = X.transpose(-1, -2)  # Transpose back
            return X
        return X


class S4DKernel(nn.Module):
    def __init__(
        self,
        d_model: int,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float = None,
    ):
        super().__init__()
        H = d_model  # Renaming for clarity from original S4 papers
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(
            dt_min
        )

        C_init = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C_init))
        self._register_param("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self._register_param("log_A_real", log_A_real, lr)
        self._register_param("A_imag", A_imag, lr)  # A_imag is not learned typically

    def forward(self, L: int):  # L is sequence length
        dt = torch.exp(self.log_dt)  # (H)
        C = _r2c(self.C)  # (H, N/2) complex
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2) complex

        dtA = A * dt.unsqueeze(-1)  # (H, N/2)

        # Kernel calculation (Convolution theorem part)
        # K_conv = C * (e^(dtA) - 1) / A
        # This is part of the HiPPO framework for state space models
        C_times_dt = C * dt.unsqueeze(
            -1
        )  # Element-wise, for HiPPO-LegS this might be different
        # For S4D, the formula using C * (exp(dtA)-1)/A is common.

        # Original S4D computes K using Vandermonde multiplication for HiPPO C_bar
        # For frequency domain kernel:
        # K_f = (C_bar * (omega - A)^-1 * B_bar) # This is for continuous case
        # Discretized version (bilinear transform or ZOH):
        # K_z = C_z * (zI - A_z)^-1 * B_z
        # The provided code uses an explicit time-domain kernel computation K

        # This K seems to be a direct computation of the impulse response
        coeffs = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H, N/2, L)
        K_complex = torch.einsum(
            "hn,hnl->hl", C * (torch.exp(dtA) - 1.0) / A, torch.exp(coeffs)
        )
        K = 2 * K_complex.real
        return K

    def _register_param(self, name: str, tensor: torch.Tensor, lr: float = None):
        if lr == 0.0:  # Treat as frozen buffer
            self.register_buffer(name, tensor)
        else:
            param = nn.Parameter(tensor)
            self.register_parameter(name, param)
            if lr is not None:  # S4-specific learning rate
                setattr(param, "_optim", {"lr": lr, "weight_decay": 0.0})


class S4D(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        **kernel_args,
    ):
        super().__init__()

        self.h_model = d_model  # H in S4 paper
        self.d_state = d_state  # N in S4 paper
        self.d_output = self.h_model  # Output dim is same as input model dim
        self.transposed = transposed  # If true, input is (B, H, L), else (B, L, H)

        self.D_skip_connection = nn.Parameter(torch.randn(self.h_model))

        self.kernel = S4DKernel(self.h_model, N=self.d_state, **kernel_args)

        # Activation and dropout after convolution and skip connection
        self.activation = nn.GELU()  # Common in S4
        self.dropout_layer = (
            DropoutNd(dropout, transposed=self.transposed)
            if dropout > 0.0
            else nn.Identity()
        )

        # Output linear GLU layer
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h_model, 2 * self.h_model, kernel_size=1),  # Project to 2*H
            nn.GLU(
                dim=1
            ),  # GLU halves the channel dimension back to H at dim=1 (channel dim for Conv1d)
        )

    def forward(self, u: torch.Tensor, **kwargs):  # u is input sequence
        if not self.transposed:  # Expects (B, H, L)
            u = u.transpose(-1, -2)

        seq_len = u.size(-1)
        k = self.kernel(L=seq_len)  # (H, L)

        # Convolution via FFT
        k_f = torch.fft.rfft(k, n=2 * seq_len)  # (H, L_fft)
        u_f = torch.fft.rfft(u.to(torch.float32), n=2 * seq_len)  # (B, H, L_fft)

        # Element-wise product in frequency domain
        y_conv_f = u_f * k_f.unsqueeze(0)  # Add batch dim to kernel_f
        y_conv = torch.fft.irfft(y_conv_f, n=2 * seq_len)[..., :seq_len]  # (B, H, L)

        # Add D skip connection (direct path for input u)
        y_skip = y_conv + u * self.D_skip_connection.unsqueeze(0).unsqueeze(
            -1
        )  # (B,H,1) broadcast

        # Activation and dropout
        y_activated = self.dropout_layer(self.activation(y_skip))

        # Output linear layer
        y_output = self.output_linear(y_activated)  # (B, H, L)

        if not self.transposed:
            y_output = y_output.transpose(-1, -2)  # (B, L, H)
        return y_output


class S4Model(BaseMILModel):  # S4MIL Wrapper
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        embed_dim: int = 512,
        s4_d_state: int = 64,  # S4 N param
        dropout_rate: float = 0.0,  # Dropout for FC and S4D
        activation: str = "gelu",  # Activation for FC
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

        fc1_layers = [nn.Linear(in_dim, embed_dim)]
        fc1_layers.append(get_activation_fn(activation))
        if dropout_rate > 0:  # Dropout after first FC's activation
            fc1_layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor_fc = nn.Sequential(*fc1_layers)

        # S4 block expects input (Batch, SeqLen, Features) if transposed=False
        # or (Batch, Features, SeqLen) if transposed=True.
        # Here, Features = embed_dim, SeqLen = num_instances.
        self.s4_block = nn.Sequential(
            nn.LayerNorm(embed_dim),  # LayerNorm before S4, applied on feature dim
            S4D(
                d_model=embed_dim,
                d_state=s4_d_state,
                dropout=dropout_rate,
                transposed=False,
            ),
        )
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.apply(initialize_weights)

    def _forward_impl(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None):
        # x: (batch_size, num_instances, in_dim) - already normalized by base class

        # Instance feature extraction
        instance_features = self.feature_extractor_fc(
            x
        )  # (batch_size, num_instances, embed_dim)

        # S4 processing: input (batch, seq_len=num_instances, features=embed_dim)
        s4_output = self.s4_block(
            instance_features
        )  # (batch_size, num_instances, embed_dim)

        # Max pooling over instances (sequence dimension)
        bag_representation, _ = torch.max(s4_output, dim=1)  # (batch_size, embed_dim)
        bag_representation = self._fuse_metadata(bag_representation, metadata)

        logits = self.classifier(bag_representation)  # (batch_size, n_classes)

        # Predictions (highest logit index)
        predictions = torch.topk(logits, 1, dim=1)[1]

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival_curves, predictions, None, {}
        else:
            probabilities = F.softmax(logits, dim=1)
            return logits, probabilities, predictions, None, {}
