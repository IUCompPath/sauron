from typing import Optional

import torch
import torch.nn as nn


def nll_surv_loss(
    hazards: torch.Tensor,
    survival: Optional[torch.Tensor],
    event_time: torch.Tensor,
    censoring_status: torch.Tensor,
    alpha: float = 0.4,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes the Negative Log-Likelihood survival loss for discrete-time survival analysis.

    This loss is composed of two parts:
    1. For uncensored samples (event observed), the loss is the negative log of the
       probability mass function P(T=t) = S(t-1) * h(t), where S is the survival
       function and h is the hazard function.
    2. For censored samples (event not observed), the loss is the negative log of the
       survival function S(t).

    An `alpha` parameter is included to weigh the uncensored loss component,
    allowing for a trade-off, as seen in some survival analysis literature.

    Args:
        hazards (torch.Tensor): Predicted hazard rates for each time interval.
            Shape: (batch_size, num_intervals).
        survival (Optional[torch.Tensor]): Pre-computed survival probabilities. If None,
            they are calculated from hazards. Shape: (batch_size, num_intervals).
        event_time (torch.Tensor): The 0-indexed time bin of the event or censoring.
            Shape: (batch_size,).
        censoring_status (torch.Tensor): Censoring status. 0 for event, 1 for censored.
            Shape: (batch_size,).
        alpha (float, optional): Weighting factor for the uncensored loss component.
            Defaults to 0.4.
        eps (float, optional): Small epsilon value to prevent log(0). Defaults to 1e-7.

    Returns:
        torch.Tensor: The mean calculated loss as a scalar tensor.
    """
    batch_size = event_time.shape[0]
    # Ensure inputs have the correct shape (batch_size, 1) for gathering
    y_true = event_time.view(batch_size, 1)
    c_status = censoring_status.view(batch_size, 1).float()

    if survival is None:
        # S(t) = product_{j=0 to t} (1 - h(j))
        survival = torch.cumprod(1 - hazards, dim=1)

    # Prepend a column of 1s to survival probabilities to represent S(-1) = 1
    # This simplifies gathering S(t-1) using y_true.
    s_padded = torch.cat([torch.ones_like(c_status), survival], dim=1)

    # For an uncensored sample at time t (y_true), likelihood is P(T=t) = h(t) * S(t-1)
    # Log-likelihood = log(h(t)) + log(S(t-1))
    log_h_y = torch.log(torch.gather(hazards, 1, y_true).clamp(min=eps))
    log_s_y_minus_1 = torch.log(torch.gather(s_padded, 1, y_true).clamp(min=eps))
    uncensored_loss = -(1 - c_status) * (log_h_y + log_s_y_minus_1)

    # For a censored sample at time t (y_true), likelihood is S(t)
    # Log-likelihood = log(S(t))
    log_s_y = torch.log(torch.gather(s_padded, 1, y_true + 1).clamp(min=eps))
    censored_loss = -c_status * log_s_y

    neg_log_likelihood = censored_loss + uncensored_loss

    # Apply alpha weighting: loss = (1 - alpha) * NLL + alpha * (uncensored part of NLL)
    loss = (1 - alpha) * neg_log_likelihood + alpha * uncensored_loss
    return loss.mean()


def ce_surv_loss(
    hazards: torch.Tensor,
    survival: Optional[torch.Tensor],
    event_time: torch.Tensor,
    censoring_status: torch.Tensor,
    alpha: float = 0.4,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes a hybrid survival loss based on Cross-Entropy.

    This loss has two components:
    1. A binary cross-entropy term that treats survival at the event/censoring time `t`
       as a binary classification problem. The "label" is the censoring status `c`
       (1 for censored/survived, 0 for event/not-survived), and the prediction is S(t).
       The loss is BCE(S(t), c).
    2. A regularization term, which is the uncensored component of the NLL loss,
       -log(P(T=t)), applied only to uncensored samples.

    The `alpha` parameter balances these two components.

    Args:
        hazards (torch.Tensor): Predicted hazard rates for each time interval.
            Shape: (batch_size, num_intervals).
        survival (Optional[torch.Tensor]): Pre-computed survival probabilities. If None,
            they are calculated from hazards. Shape: (batch_size, num_intervals).
        event_time (torch.Tensor): The 0-indexed time bin of the event or censoring.
            Shape: (batch_size,).
        censoring_status (torch.Tensor): Censoring status. 0 for event, 1 for censored.
            Shape: (batch_size,).
        alpha (float, optional): Weighting factor for the regularization term.
            Defaults to 0.4.
        eps (float, optional): Small epsilon value to prevent log(0). Defaults to 1e-7.

    Returns:
        torch.Tensor: The mean calculated loss as a scalar tensor.
    """
    batch_size = event_time.shape[0]
    y_true = event_time.view(batch_size, 1)
    c_status = censoring_status.view(batch_size, 1).float()

    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)

    # Main loss component: Binary Cross-Entropy on S(t)
    s_y = torch.gather(survival, 1, y_true).clamp(min=eps, max=1 - eps)
    bce_loss = -c_status * torch.log(s_y) - (1 - c_status) * torch.log(1 - s_y)

    # Regularization term: Uncensored part of the NLL loss -log(h(t)) - log(S(t-1))
    s_padded = torch.cat([torch.ones_like(c_status), survival], dim=1)
    log_h_y = torch.log(torch.gather(hazards, 1, y_true).clamp(min=eps))
    log_s_y_minus_1 = torch.log(torch.gather(s_padded, 1, y_true).clamp(min=eps))
    nll_uncensored = -(1 - c_status) * (log_h_y + log_s_y_minus_1)

    loss = (1 - alpha) * bce_loss + alpha * nll_uncensored
    return loss.mean()


class NLLSurvLoss(nn.Module):
    """
    Computes the Negative Log-Likelihood survival loss (see `nll_surv_loss`).
    This class wrapper allows setting a default `alpha` value during initialization.

    Args:
        alpha (float, optional): Default weighting factor for the uncensored loss
            component. Defaults to 0.15.
    """

    def __init__(self, alpha: float = 0.15):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        hazards: torch.Tensor,
        survival: Optional[torch.Tensor],
        event_time: torch.Tensor,
        censoring_status: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """Calculates the loss, overriding the default alpha if provided."""
        current_alpha = alpha if alpha is not None else self.alpha
        return nll_surv_loss(
            hazards=hazards,
            survival=survival,
            event_time=event_time,
            censoring_status=censoring_status,
            alpha=current_alpha,
        )


class CrossEntropySurvLoss(nn.Module):
    """
    Computes the hybrid Cross-Entropy survival loss (see `ce_surv_loss`).
    This class wrapper allows setting a default `alpha` value during initialization.

    Args:
        alpha (float, optional): Default weighting factor for the regularization
            term. Defaults to 0.15.
    """

    def __init__(self, alpha: float = 0.15):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        hazards: torch.Tensor,
        survival: Optional[torch.Tensor],
        event_time: torch.Tensor,
        censoring_status: torch.Tensor,
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """Calculates the loss, overriding the default alpha if provided."""
        current_alpha = alpha if alpha is not None else self.alpha
        return ce_surv_loss(
            hazards=hazards,
            survival=survival,
            event_time=event_time,
            censoring_status=censoring_status,
            alpha=current_alpha,
        )


class CoxSurvLoss(nn.Module):
    """
    Computes the Cox Proportional Hazards loss.

    This implementation uses the negative partial log-likelihood. It relies on
    efficient tensor broadcasting to compute the risk sets, avoiding slow loops.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        hazards: torch.Tensor,
        event_times: torch.Tensor,
        censoring_status: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the Cox loss.

        Args:
            hazards (torch.Tensor): Predicted log-risk scores (theta) for each patient.
                Shape: (batch_size, 1) or (batch_size,).
            event_times (torch.Tensor): Observed event or censoring times.
                Shape: (batch_size, 1) or (batch_size,).
            censoring_status (torch.Tensor): Censoring status, where 0 indicates an
                event and 1 indicates censoring.
                Shape: (batch_size, 1) or (batch_size,).

        Returns:
            torch.Tensor: The mean negative partial log-likelihood as a scalar tensor.
        """
        # Squeeze inputs to ensure they are 1D for risk set computation.
        theta = hazards.squeeze(dim=-1)
        times = event_times.squeeze(dim=-1)
        c_status = censoring_status.squeeze(dim=-1)

        # The risk set for an individual 'i' includes all individuals 'j'
        # whose event/censoring time is at or after individual 'i's time.
        risk_set_matrix = (times.unsqueeze(0) >= times.unsqueeze(1)).float()
        risk_set_matrix = risk_set_matrix.to(theta.device)

        # Denominator of the partial likelihood: log-sum-exp over the risk set.
        exp_theta = torch.exp(theta)
        log_risk_sum = torch.log(torch.sum(exp_theta * risk_set_matrix, dim=1) + 1e-9)

        # Partial log-likelihood is summed over uncensored individuals:
        # L_i = theta_i - log(sum_{j in R_i} exp(theta_j))
        # We use (1 - c_status) as a mask for uncensored events.
        loss = -torch.mean((theta - log_risk_sum) * (1 - c_status))
        return loss
