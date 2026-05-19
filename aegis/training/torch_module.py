import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import MetricCollection

from aegis.losses.surv_loss import CoxSurvLoss, NLLSurvLoss
from aegis.mil_models.models_factory import mil_model_factory


class AegisTorchModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = mil_model_factory(args)

        # --- Optimization 1: Clean Loss & Metric Initialization ---
        if self.args.task_type.lower() == "classification":
            self.loss_fn = nn.CrossEntropyLoss()

            # Use MetricCollection to group metrics
            metrics = MetricCollection(
                {
                    "auc": torchmetrics.AUROC(
                        task="multiclass", num_classes=args.n_classes
                    ),
                    "acc": torchmetrics.Accuracy(
                        task="multiclass", num_classes=args.n_classes
                    ),
                }
            )

            # Clone for different stages to maintain separate states
            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
            self.test_metrics = metrics.clone(prefix="test_")

        elif self.args.task_type.lower() == "survival":
            if self.args.bag_loss == "nll_surv":
                self.loss_fn = NLLSurvLoss(alpha=self.args.alpha_surv)
            elif self.args.bag_loss == "cox_surv":
                self.loss_fn = CoxSurvLoss()
            else:
                raise ValueError(f"Unknown survival loss: {self.args.bag_loss}")
        else:
            raise ValueError(f"Unknown task type: {self.args.task_type}")

    def forward(self, x):
        return self.model(x)

    # --- Optimization 2: Cleaner Helper returning Dictionary ---
    def compute_step(self, batch):
        results = {}

        if self.args.task_type.lower() == "classification":
            data, label = batch[0], batch[1]  # Robust to 2 or 3 item batches
            logits, probs, preds, _, _ = self.model(data)
            loss = self.loss_fn(logits, label)

            results.update({"loss": loss, "probs": probs, "label": label})

        elif self.args.task_type.lower() == "survival":
            data, label, event, c = batch
            hazards, S, preds, _, _ = self.model(data)

            loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=c)
            risk = -torch.sum(S, dim=1)

            results.update({"loss": loss, "risk": risk, "event": event, "c": c})

        return results
