import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Args:
        alpha (float, list, torch.Tensor, optional): Weights for each class.
            If float, it's treated as the weight for the rare class (binary).
            If list/Tensor, it should have length equal to number of classes.
        gamma (float): Focusing parameter. Higher values focus more on hard examples. Default: 2.0.
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'.
        label_smoothing (float): Label smoothing factor (0.0 to 1.0). Default: 0.0.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # OPTIMIZATION: Register alpha as a buffer.
        # This ensures it automatically moves to GPU when you call model.cuda()
        # and is saved with the model checkpoint.
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # 1. Compute standard Cross Entropy (includes alpha-weighting & smoothing)
        # We use reduction='none' so we can apply the focal weight per-sample first
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
        )

        # 2. Compute the Focal Term (1 - pt)
        # We detach pt here usually to stop gradients flowing back through the weight itself,
        # but keeping it attached is also valid for "hard" focal loss.
        pt = torch.exp(-ce_loss)  # Mathematical trick: exp(-CE) is effectively pt
        focal_weight = (1 - pt) ** self.gamma

        # 3. Combine
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class Poly1Loss(nn.Module):
    """
    Poly1Loss: A alternative to Focal Loss that adds a polynomial term to Cross Entropy.
    Often provides better stability and accuracy than Focal Loss by preventing
    over-suppression of the gradient for easy samples.

    Formula: L_Poly1 = L_CE + epsilon * (1 - Pt)

    Args:
        num_classes (int): Number of classes.
        epsilon (float): Weight for the polynomial term. Default: 1.0.
        reduction (str): 'mean', 'sum', or 'none'.
        weight (torch.Tensor, optional): Class weights.
    """

    def __init__(self, num_classes, epsilon=1.0, reduction="mean", weight=None):
        super(Poly1Loss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction

        # OPTIMIZATION: Register weight as buffer
        if weight is not None:
            if isinstance(weight, (list, tuple)):
                weight = torch.tensor(weight, dtype=torch.float32)
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, inputs, targets):
        # 1. Cross Entropy (Base Loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight)

        # 2. Polynomial Term (Gradient Booster)
        pt = F.softmax(inputs, dim=-1)
        p_target = pt.gather(1, targets.view(-1, 1)).squeeze(1)

        # The Poly1 term: ε * (1 - Pt)
        # This pushes the model to not just get the class "mostly right" (Pt=0.6)
        # but "completely right" (Pt=1.0)
        poly1 = self.epsilon * (1 - p_target)

        loss = ce_loss + poly1

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
