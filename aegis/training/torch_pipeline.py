import os
from typing import Tuple
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sksurv.metrics import concordance_index_censored

from aegis.data.data_utils import get_dataloader
from aegis.training.torch_module import AegisTorchModule
from aegis.utils.optimizers import get_optim


class EarlyStopping:
    def __init__(self, patience=20, mode="max", verbose=True):
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "max" and score < self.best_score) or (
            self.mode == "min" and score > self.best_score
        ):
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train_fold(
    train_dataset,
    val_dataset,
    test_dataset,
    cur_fold_num: int,
    args,
    experiment_base_results_dir: str,
) -> Tuple:
    """
    Trains and evaluates a model for a single fold using pure PyTorch.
    """
    print(f"Initializing pure torch training for fold {cur_fold_num}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders
    n_subsamples = getattr(args, "n_subsamples", None)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn_type": args.task_type,
        "n_subsamples": n_subsamples,
        "num_workers": getattr(args, "num_workers", 8),
        "pin_memory": True,
        "persistent_workers": True if getattr(args, "num_workers", 8) > 0 else False,
    }

    train_loader = get_dataloader(
        train_dataset,
        shuffle=True,
        use_weighted_sampler=args.weighted_sample,
        **loader_kwargs,
    )
    val_loader = get_dataloader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = get_dataloader(test_dataset, shuffle=False, **loader_kwargs)

    # Model
    model = AegisTorchModule(args).to(device)

    # Optimizer & Scheduler
    # Pass the inner model to get_optim as it likely expects the model architecture
    optimizer = get_optim(model.model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    # Logging & Checkpoints
    checkpoint_dir = os.path.join(experiment_base_results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=experiment_base_results_dir)

    monitor_metric = "val_c_index" if args.task_type == "survival" else "val_auc"
    monitor_mode = "max" if monitor_metric in ["val_c_index", "val_auc"] else "min"

    early_stopping = EarlyStopping(patience=20, mode=monitor_mode, verbose=True)
    best_metric = -np.inf if monitor_mode == "max" else np.inf
    best_model_path = os.path.join(checkpoint_dir, f"s_{cur_fold_num}_best.pt")

    # Training Loop
    for epoch in range(args.max_epochs):
        # --- Train ---
        model.train()
        if args.task_type == "classification":
            model.train_metrics.reset()

        train_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"Fold {cur_fold_num} Epoch {epoch} Train", leave=False
        )

        for batch in pbar:
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]

            optimizer.zero_grad()
            res = model.compute_step(batch)
            loss = res["loss"]
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

            if args.task_type == "classification":
                model.train_metrics.update(res["probs"], res["label"])

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("train_loss", avg_train_loss, epoch)

        if args.task_type == "classification":
            metrics = model.train_metrics.compute()
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)

        # --- Validation ---
        model.eval()
        if args.task_type == "classification":
            model.val_metrics.reset()

        val_loss = 0.0
        val_risks, val_events, val_cs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
                ]
                res = model.compute_step(batch)
                val_loss += res["loss"].item()

                if args.task_type == "classification":
                    model.val_metrics.update(res["probs"], res["label"])
                elif args.task_type == "survival":
                    val_risks.append(res["risk"].cpu())
                    val_events.append(res["event"].cpu())
                    val_cs.append(res["c"].cpu())

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("val_loss", avg_val_loss, epoch)

        current_metric = 0.0
        if args.task_type == "classification":
            metrics = model.val_metrics.compute()
            for k, v in metrics.items():
                writer.add_scalar(k, v, epoch)
            current_metric = metrics.get("val_auc", 0.0).item()
        elif args.task_type == "survival":
            if val_risks:
                risks = torch.cat(val_risks).numpy()
                events = torch.cat(val_events).numpy()
                cs = torch.cat(val_cs).numpy()
                event_observed = (1 - cs).astype(bool)
                try:
                    c_index = concordance_index_censored(event_observed, events, risks)[
                        0
                    ]
                    writer.add_scalar("val_c_index", c_index, epoch)
                    current_metric = c_index
                except Exception as e:
                    print(f"Error computing C-Index: {e}")
                    current_metric = 0.0

        print(
            f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, {monitor_metric} {current_metric:.4f}"
        )

        # --- Checkpoint ---
        save = False
        if monitor_mode == "max":
            if current_metric > best_metric:
                best_metric = current_metric
                save = True
        else:
            if current_metric < best_metric:
                best_metric = current_metric
                save = True

        if save:
            print(f"Saving best model to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)

        # --- Early Stopping ---
        early_stopping(current_metric)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # --- Final Evaluation on Best Model ---
    print(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    def evaluate_set(loader, prefix="test"):
        if args.task_type == "classification":
            # Use test_metrics for consistency or clone a new one
            metrics_collection = model.test_metrics.clone(prefix=f"{prefix}_")
            metrics_collection.reset()

        risks, events, cs = [], [], []

        with torch.no_grad():
            for batch in loader:
                batch = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
                ]
                res = model.compute_step(batch)

                if args.task_type == "classification":
                    metrics_collection.update(res["probs"], res["label"])
                elif args.task_type == "survival":
                    risks.append(res["risk"].cpu())
                    events.append(res["event"].cpu())
                    cs.append(res["c"].cpu())

        results = {}
        if args.task_type == "classification":
            computed = metrics_collection.compute()
            results.update({k: v.item() for k, v in computed.items()})
        elif args.task_type == "survival":
            if risks:
                r = torch.cat(risks).numpy()
                e = torch.cat(events).numpy()
                c = torch.cat(cs).numpy()
                e_obs = (1 - c).astype(bool)
                try:
                    c_idx = concordance_index_censored(e_obs, e, r)[0]
                    results[f"{prefix}_c_index"] = c_idx
                except Exception:
                    results[f"{prefix}_c_index"] = 0.0

        return results

    # Evaluate on Val (again, for return values) and Test
    val_results = evaluate_set(
        val_loader, prefix="test"
    )  # Pipeline uses 'test' prefix for val results return
    test_results = evaluate_set(test_loader, prefix="test")

    writer.close()

    # Extract metrics for return
    if args.task_type == "classification":
        val_auc = val_results.get("test_auc", 0.0)
        val_acc = val_results.get("test_acc", 0.0)
        test_auc = test_results.get("test_auc", 0.0)
        test_acc = test_results.get("test_acc", 0.0)
        return (
            {},
            test_auc,
            val_auc,
            test_acc,
            val_acc,
        )
    elif args.task_type == "survival":
        val_c_index = val_results.get("test_c_index", 0.0)
        test_c_index = test_results.get("test_c_index", 0.0)
        return {}, test_c_index, val_c_index

    return {}, 0.0, 0.0
