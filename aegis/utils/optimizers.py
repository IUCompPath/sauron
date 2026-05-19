from typing import Any

import torch.nn as nn
import torch.optim as optim


def get_optim(model: nn.Module, args: Any) -> optim.Optimizer:
    if args.opt == "adam":
        return optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.reg,
        )
    elif args.opt == "adamw":
        return optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.reg,
        )
    elif args.opt == "sgd":
        return optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.reg,
        )
    raise NotImplementedError(f"Optimizer {args.opt} not implemented")
