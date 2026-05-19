import torch.nn as nn


def get_activation_fn(activation_name: str) -> nn.Module:
    """Returns the activation function module."""
    if activation_name.lower() == "relu":
        return nn.ReLU()
    elif activation_name.lower() == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")
