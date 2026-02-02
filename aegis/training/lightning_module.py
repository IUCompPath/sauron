import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from sksurv.metrics import concordance_index_censored
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection

from aegis.losses.surv_loss import CoxSurvLoss, NLLSurvLoss
from aegis.mil_models.models_factory import mil_model_factory
from aegis.utils.optimizers import get_optim


class aegis(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.model = mil_model_factory(args)

        # Storage for epoch-end aggregation (Fixes PL 2.0+ deprecation)
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # --- Optimization 1: Clean Loss & Metric Initialization ---
        if self.args.task_type.lower() == "classification":
            if getattr(self.args, "loss_type", "focal") == "poly":
                from aegis.losses.classification_loss import Poly1Loss

                # Using Poly1Loss with epsilon=1.0
                self.loss_fn = Poly1Loss(num_classes=args.n_classes, epsilon=1.0)
            else:
                from aegis.losses.classification_loss import FocalLoss

                # Using Focal Loss with Label Smoothing (0.1) and Gamma (2.0) to handle class imbalance and overfitting
                self.loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)

            # Use MetricCollection to group metrics
            # LIGHTWEIGHT metrics for Training
            self.train_metrics = MetricCollection(
                {
                    "bal_acc": torchmetrics.BalancedAccuracy(
                        task="multiclass", num_classes=args.n_classes
                    )
                },
                prefix="train_",
            )

            # HEAVY metrics for Validation (Keep AUROC here)
            self.val_metrics = MetricCollection(
                {
                    "auc": torchmetrics.AUROC(
                        task="multiclass", num_classes=args.n_classes
                    ),
                    "bal_acc": torchmetrics.BalancedAccuracy(
                        task="multiclass", num_classes=args.n_classes
                    ),
                },
                prefix="val_",
            )

            self.test_metrics = MetricCollection(
                {
                    "auc": torchmetrics.AUROC(
                        task="multiclass", num_classes=args.n_classes
                    ),
                    "bal_acc": torchmetrics.BalancedAccuracy(
                        task="multiclass", num_classes=args.n_classes
                    ),
                },
                prefix="test_",
            )

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
    def _get_outputs_and_loss(self, batch):
        results = {}

        if self.args.task_type.lower() == "classification":
            data, label = batch[0], batch[1]
            logits, probs, preds, _, _ = self.model(data)
            loss = self.loss_fn(logits, label)

            results.update({"loss": loss, "probs": probs, "label": label})

        elif self.args.task_type.lower() == "survival":
            data, label, event, c = batch
            hazards, S, preds, _, _ = self.model(data)
            loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=c)
            risk = -torch.sum(S, dim=1).detach()

            results.update({"loss": loss, "risk": risk, "event": event, "c": c})

        return results

    def training_step(self, batch, batch_idx):
        res = self._get_outputs_and_loss(batch)
        loss = res["loss"]

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch[0].size(0),
        )

        if self.args.task_type.lower() == "classification":
            self.train_metrics.update(res["probs"].detach(), res["label"])

        return loss

    def on_train_epoch_end(self):
        if self.args.task_type.lower() == "classification":
            output = self.train_metrics.compute()
            self.log_dict(output, prog_bar=True)
            self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        res = self._get_outputs_and_loss(batch)
        self.log("val_loss", res["loss"], prog_bar=True, batch_size=batch[0].size(0))

        if self.args.task_type.lower() == "classification":
            self.val_metrics.update(res["probs"], res["label"])
        elif self.args.task_type.lower() == "survival":
            # Store tensors, don't convert to numpy yet (Optimization 3)
            self.validation_step_outputs.append(
                {
                    "risk": res["risk"].detach().cpu(),
                    "event": res["event"].detach().cpu(),
                    "c": res["c"].detach().cpu(),
                }
            )

    def on_validation_epoch_end(self):
        if self.args.task_type.lower() == "classification":
            output = self.val_metrics.compute()
            self.log_dict(output, prog_bar=True)
            self.val_metrics.reset()

        elif self.args.task_type.lower() == "survival":
            # Compute C-Index efficiently
            if self.validation_step_outputs:
                risks = torch.cat(
                    [x["risk"] for x in self.validation_step_outputs]
                ).numpy()
                events = torch.cat(
                    [x["event"] for x in self.validation_step_outputs]
                ).numpy()
                cs = torch.cat([x["c"] for x in self.validation_step_outputs]).numpy()

                event_observed = (1 - cs).astype(bool)
                try:
                    c_index = concordance_index_censored(event_observed, events, risks)[
                        0
                    ]
                    self.log("val_c_index", c_index, prog_bar=True)
                except Exception as e:
                    print(f"Error computing C-Index: {e}")

                self.validation_step_outputs.clear()  # Free memory

    def test_step(self, batch, batch_idx):
        res = self._get_outputs_and_loss(batch)
        self.log("test_loss", res["loss"], batch_size=batch[0].size(0))

        if self.args.task_type.lower() == "classification":
            self.test_metrics.update(res["probs"], res["label"])
        elif self.args.task_type.lower() == "survival":
            self.test_step_outputs.append(
                {
                    "risk": res["risk"].detach().cpu(),
                    "event": res["event"].detach().cpu(),
                    "c": res["c"].detach().cpu(),
                }
            )

    def on_test_epoch_end(self):
        if self.args.task_type.lower() == "classification":
            output = self.test_metrics.compute()
            self.log_dict(output)
            self.test_metrics.reset()

        elif self.args.task_type.lower() == "survival":
            if self.test_step_outputs:
                risks = torch.cat([x["risk"] for x in self.test_step_outputs]).numpy()
                events = torch.cat([x["event"] for x in self.test_step_outputs]).numpy()
                cs = torch.cat([x["c"] for x in self.test_step_outputs]).numpy()

                event_observed = (1 - cs).astype(bool)
                c_index = concordance_index_censored(event_observed, events, risks)[0]
                self.log("test_c_index", c_index)

                self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = get_optim(self.model, self.args)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.max_epochs)
        return [optimizer], [scheduler]
