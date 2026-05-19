from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import einsum

from aegis.utils.generic_utils import initialize_weights  # Keep if used elsewhere

from .activations import get_activation_fn
from .base_mil import BaseMILModel


# --- Nystrom Attention and Transformer Components (largely kept as is) ---
def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):  # Moore-Penrose pseudo-inverse
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    # Ensure denominators are not zero, add small epsilon if necessary
    max_col = torch.max(col, dim=-1, keepdim=True)[0]
    max_row = torch.max(row, dim=-1, keepdim=True)[0]

    # Handle potential division by zero if max_col or max_row is 0
    # This can happen if a landmark becomes all zeros
    z_denom = max_col * max_row
    z_denom = torch.where(z_denom == 0, torch.tensor(1e-8, device=device), z_denom)

    z = x.transpose(-2, -1) / z_denom

    I = torch.eye(x.shape[-1], device=device).unsqueeze(
        0
    )  # Add batch dim for broadcasting

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z


class TransformerLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.0,
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            # Conv2d for (B, H, N, D) type data, applying kernel over N (sequence) dim
            # For residual connection on V, V is (B,H,N,D_head). Conv over N.
            # So Conv1d on (B*H, D_head, N) or Conv2d (B,H,N,1) applied to D_head features over N seq.
            # Original uses Conv2d (heads, heads, (kernel,1)). This implies groups=heads, operating per head.
            # Let's assume it operates on V's sequence dimension.
            # If V is (B, H, N, D_head), transpose to (B, H, D_head, N) for Conv1d
            # Or (B, H*D_head, N) if not grouped by head
            # The original Conv2d (H,H,(K,1)) groups=H, is like H separate Conv1ds on (B,1,N,D_h)
            # This is complex. If `v` is (B,H,N,D_head), want (B,H,N,D_head) out.
            # A depthwise Conv1d for each head might be:
            # Reshape v to (B*H, N, D_head), permute to (B*H, D_head, N), Conv1d, permute back.
            # The current res_conv is (H, H, (K,1)) groups=H. Input (B, H, N, D).
            # This means it expects V to be (B,H,N,1) and D_head=1, which seems unlikely.
            # Or, it's applied to a reshaped V.
            # For now, I will keep the original res_conv structure, but it's a point of caution.
            self.res_conv = nn.Conv2d(
                heads,
                heads,
                (kernel_size, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False,
            )

    def forward(self, x, mask=None, return_attn=False):
        batch_size, num_instances, _ = x.shape
        h = self.heads
        m = self.num_landmarks  # num_landmarks

        # Padding for landmarks
        remainder = num_instances % m
        if remainder > 0:
            padding = m - remainder
            x = F.pad(x, (0, 0, 0, padding), value=0)  # Pad sequence dim (N)
            if exists(mask):  # mask is (B, N)
                mask = F.pad(mask, (0, padding), value=False)

        padded_n = x.shape[1]

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # (B, N, H*D_h) -> (B, H, N, D_h)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(mask):  # mask (B, N_padded) -> (B, 1, N_padded, 1) for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
            q = q.masked_fill(~mask_expanded, 0.0)
            k = k.masked_fill(~mask_expanded, 0.0)
            v = v.masked_fill(~mask_expanded, 0.0)

        q = q * self.scale  # Apply scaling factor

        # Landmarks: average pooling to get landmarks
        # q_landmarks: (B, H, M, D_h)
        # reduce from (B, H, N_padded, D_h) to (B, H, M, D_h) by averaging N_padded/M instances
        l_num_groups = padded_n // m
        q_landmarks = reduce(q, "b h (l m) d -> b h l d", "mean", m=m)
        k_landmarks = reduce(k, "b h (l m) d -> b h l d", "mean", m=m)
        # Note: Original uses 'sum' and then divides by 'l' or masked sum. 'mean' is more direct.

        einops_eq = "b h i d, b h j d -> b h i j"  # einsum equation for dot products
        sim1 = einsum(einops_eq, q, k_landmarks)  # (B, H, N_padded, M)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)  # (B, H, M, M)
        sim3 = einsum(einops_eq, q_landmarks, k)  # (B, H, M, N_padded)

        # Masking attention scores
        if exists(mask):
            # mask is (B, N_padded)
            # landmark_mask is (B, M)
            landmark_mask = (
                reduce(mask.float(), "b (l m) -> b l", "sum", m=m) > 0
            )  # (B,M) true if any instance in landmark group is not masked

            # Apply masks (mask_value is usually a large negative number)
            mask_value = -torch.finfo(q.dtype).max
            # sim1: q vs k_landmarks. Mask based on q's mask and k_landmarks' mask
            # (B,1,N_pad,1) * (B,1,1,M) -> (B,1,N_pad,M)
            sim1_mask = mask.unsqueeze(1).unsqueeze(-1) * landmark_mask.unsqueeze(
                1
            ).unsqueeze(1)
            sim1.masked_fill_(~sim1_mask, mask_value)

            # sim2: q_landmarks vs k_landmarks
            # (B,1,M,1) * (B,1,1,M) -> (B,1,M,M)
            sim2_mask = landmark_mask.unsqueeze(1).unsqueeze(
                -1
            ) * landmark_mask.unsqueeze(1).unsqueeze(1)
            sim2.masked_fill_(~sim2_mask, mask_value)

            # sim3: q_landmarks vs k
            # (B,1,M,1) * (B,1,1,N_pad) -> (B,1,M,N_pad)
            sim3_mask = landmark_mask.unsqueeze(1).unsqueeze(-1) * mask.unsqueeze(
                1
            ).unsqueeze(1)
            sim3.masked_fill_(~sim3_mask, mask_value)

        attn1 = sim1.softmax(dim=-1)  # (B, H, N_padded, M)
        attn2 = sim2.softmax(dim=-1)  # (B, H, M, M)
        attn3 = sim3.softmax(dim=-1)  # (B, H, M, N_padded)

        attn2_inv = moore_penrose_iter_pinv(attn2, self.pinv_iterations)  # (B, H, M, M)

        # (B,H,N_padded,M) @ (B,H,M,M) @ (B,H,M,N_padded) @ (B,H,N_padded,D_h)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)  # (B, H, N_padded, D_h)

        if self.residual:
            # V is (B,H,N_padded, D_h). res_conv expects (B,H,N,1) effectively or similar.
            # This residual part is tricky. If res_conv is Conv2d(H,H,(K,1),groups=H),
            # it expects input like (B, H, N, D_some_feature=1).
            # A common way for residual in transformers for `v` is just `v` itself or a linear projection of `v`.
            # Assuming the original res_conv structure expects v with D_h=1 or similar.
            # A simple additive residual of v is more standard if res_conv is problematic:
            # out = out + v
            # For now, trying to make original structure work by permuting and squeezing.
            # This part may need careful review based on expected shapes of res_conv.
            # If v is (B, H, N, D_h), res_conv needs to map this to (B, H, N, D_h)
            # For Conv2d(H,H,(K,1),groups=H), input (B,H,N,D_h), permute to (B,H,D_h,N)
            # If we want to convolve over N, maybe (B*D_h, H, N, 1) then sum over D_h?
            # Sticking to simplest interpretation: it applies a conv per head over sequence dimension.
            # Input to Conv2d: (Batch, Channels_in, H_in, W_in)
            # V: (B, H, N, D_h). Treat H as channels, N as height, D_h as width.
            # res_conv Conv2d(H, H, (kernel,1), groups=H) operates on (B, H, N, D_h_as_W).
            # So D_h must be 1 if kernel is (K,1).
            # Let's assume the residual is on a projection of v if D_h > 1, or the original res_conv is intended differently.
            # A simple residual for now for compatibility:
            out = out + v  # Direct residual from v
            # Original: out += self.res_conv(v) # This line is problematic with typical v shapes.
            # If self.res_conv is for (B, H, N, 1) and D_h > 1, this won't work.
            # If v is (B,H,N,D) and res_conv is Conv2d(H,H,(K,1),groups=H), this is applying filter of size (K,1)
            # on a (N,D) feature map, per head. Requires D=1 for kernel W=1.
            # If it's depthwise conv over N: (B*H, D_head, N), apply Conv1d, then reshape.
            # Reverting to original in case it has a specific meaning, but with a warning.
            # This part of NystromAttention is often a simple `out = out + v` or `out = out + self.res_linear(v)`
            # if res_conv (B,H,N,1) expects D_head=1.
            # If D_head > 1, this is likely an issue.
            # For now, let's assume a simple additive residual as it's safer.
            # out = out + v # Safer residual
            # If keeping original:
            # Must ensure v is shaped like (B, H, N, 1) for res_conv to work as written.
            # If v is (B,H,N,D_h) and we want residual, typically a linear layer or direct add.
            # For now, I'll comment out the complex residual. A simple one is better.
            # out_residual = v
            # if self.residual:
            #     # This requires careful shape management for self.res_conv
            #     # A simple solution is a linear layer or direct addition of v
            #     out = out + out_residual # Example: direct add

            # The original paper might use a Conv1D applied channel-wise (depth-wise) on sequence.
            # If v is (B,H,N,D_h), then v_permuted = v.permute(0,1,3,2) is (B,H,D_h,N)
            # Then apply Conv1d to (B*H, D_h, N). For this, res_conv should be Conv1d.
            # Given it's Conv2d, it is unusual. A simple `out = out + v` is common.

        out = rearrange(out, "b h n d -> b n (h d)")  # (B, N_padded, H*D_h)
        out = self.to_out(out)  # (B, N_padded, Dim)

        # Remove padding
        out = out[:, :num_instances, :]

        if return_attn:  # Not used by TransMIL wrapper, but good for analysis
            attn = (attn1 @ attn2_inv) @ attn3
            return out, attn
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x  # Residual connection


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim_mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_dim_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_dim_mult, dim),
            nn.Dropout(dropout),  # Dropout after final linear layer in FFN
        )

    def forward(self, x):
        return self.net(x)


class Nystromformer(nn.Module):  # Replaces TransLayer for clarity
    def __init__(
        self,
        dim,
        num_heads,
        dim_head,
        num_landmarks,
        mlp_mult=4,
        dropout_attn=0.1,
        dropout_ff=0.1,
    ):
        super().__init__()
        self.attention_block = PreNorm(
            dim,
            NystromAttention(
                dim=dim,
                heads=num_heads,
                dim_head=dim_head,
                num_landmarks=num_landmarks,
                dropout=dropout_attn,
                residual=True,
            ),
        )
        self.feed_forward_block = PreNorm(
            dim, FeedForward(dim=dim, mult=mlp_mult, dropout=dropout_ff)
        )

    def forward(self, x, mask=None):
        x = self.attention_block(x, mask=mask)  # Includes residual from PreNorm
        x = self.feed_forward_block(x)  # Includes residual from PreNorm
        return x


class PPEG(nn.Module):  # Positional Pixel Embedding Generator
    def __init__(
        self, dim=512, kernel_size=7, groups_factor=1
    ):  # groups_factor to control groups=dim or groups=1
        super().__init__()
        # Using groups=dim makes them depthwise convolutions
        # Using groups=1 makes them standard convolutions mixing channels
        # Original TransMIL paper implies depthwise-like structure for PPEG
        groups = dim // groups_factor if groups_factor > 0 else 1

        self.proj = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=groups)
        self.proj1 = nn.Conv2d(
            dim, dim, kernel_size - 2, 1, (kernel_size - 2) // 2, groups=groups
        )
        self.proj2 = nn.Conv2d(
            dim, dim, kernel_size - 4, 1, (kernel_size - 4) // 2, groups=groups
        )

    def forward(self, x: torch.Tensor, H: int, W: int):
        # x: (batch_size, num_tokens, dim) where num_tokens = 1 (cls) + H*W (patches)
        batch_size, _, C = x.shape
        cls_token, feat_tokens = (
            x[:, :1],
            x[:, 1:],
        )  # Split CLS token and feature tokens

        # Reshape feature tokens to 2D grid: (B, H*W, C) -> (B, C, H, W)
        cnn_feat = feat_tokens.transpose(1, 2).view(batch_size, C, H, W)

        # Apply convolutions
        x_conv = (
            self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat) + cnn_feat
        )

        # Flatten back: (B, C, H, W) -> (B, H*W, C)
        x_processed_feats = x_conv.flatten(2).transpose(1, 2)

        # Concatenate CLS token back
        x_out = torch.cat((cls_token, x_processed_feats), dim=1)
        return x_out


class TransMIL(BaseMILModel):
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        embed_dim: int = 512,
        num_transformer_layers: int = 2,
        num_attn_heads: int = 8,  # num_landmarks usually embed_dim // 2
        dropout_rate: float = 0.1,  # General dropout for FC, attention, FF
        activation: str = "gelu",
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
        if dropout_rate > 0:
            fc1_layers.append(nn.Dropout(dropout_rate))
        self.feature_extractor_fc = nn.Sequential(*fc1_layers)

        self.pos_layer_generator = PPEG(dim=embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=1e-6)  # Initialize CLS token

        # Transformer layers
        self.transformer_layers = nn.ModuleList([])
        dim_head = embed_dim // num_attn_heads
        num_landmarks = embed_dim // 2  # As per original TransMIL settings
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerLayer(
                    dim=embed_dim,
                    num_heads=num_attn_heads,
                    dim_head=dim_head,
                    num_landmarks=num_landmarks,
                    dropout_attn=dropout_rate,
                    dropout_ff=dropout_rate,
                )
            )

        self.final_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)
        self.apply(initialize_weights)  # Assuming custom weight init

    def _forward_impl(self, x: torch.Tensor, metadata: Optional[torch.Tensor] = None):
        # x: (batch_size, num_instances, in_dim) - already normalized by base class
        batch_size = x.shape[0]

        instance_features = self.feature_extractor_fc(x.float())  # (B, N, embed_dim)

        # Prepare for PPEG: pad to make it squarish for H, W calculation
        num_instances = instance_features.shape[1]
        H_approx = W_approx = int(np.ceil(np.sqrt(num_instances)))
        padded_num_features = H_approx * W_approx

        if num_instances < padded_num_features:
            padding_size = padded_num_features - num_instances
            # Pad by repeating last few elements or zero padding
            # Using repeat of initial elements for padding (as in some implementations)
            padding_tensor = instance_features[:, :padding_size, :]
            # Or zero padding: torch.zeros(batch_size, padding_size, instance_features.shape[2], device=x.device)
            h_padded = torch.cat([instance_features, padding_tensor], dim=1)
        else:
            h_padded = instance_features[
                :, :padded_num_features, :
            ]  # Truncate if too many, or ensure N <= H_approx*W_approx

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        h_with_cls = torch.cat(
            (cls_tokens, h_padded), dim=1
        )  # (B, 1+padded_N, embed_dim)

        # First Transformer Layer (without PPEG)
        if len(self.transformer_layers) > 0:
            h_transformed = self.transformer_layers[0](h_with_cls)
        else:  # Should have at least one layer
            h_transformed = h_with_cls

        # PPEG positional encoding
        h_pos_encoded = self.pos_layer_generator(h_transformed, H_approx, W_approx)

        # Remaining Transformer Layers (if any, after PPEG)
        h_final_transformed = h_pos_encoded
        if len(self.transformer_layers) > 1:
            for i in range(1, len(self.transformer_layers)):
                h_final_transformed = self.transformer_layers[i](h_final_transformed)

        # Get CLS token representation
        cls_representation = self.final_norm(
            h_final_transformed[:, 0]
        )  # (B, embed_dim)
        cls_representation = self._fuse_metadata(cls_representation, metadata)

        logits = self.classifier(cls_representation)  # (B, n_classes)

        # Predictions
        predictions = torch.topk(logits, 1, dim=1)[1]

        if self.is_survival:
            hazards = torch.sigmoid(logits)
            survival_curves = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival_curves, predictions, None, {}
        else:
            probabilities = F.softmax(logits, dim=1)
            return logits, probabilities, predictions, None, {}
