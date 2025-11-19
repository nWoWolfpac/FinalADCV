# src/utils.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


#############################################
# NEW: unified loader for image data
#############################################
def _get_images_from_batch(batch, device):
    """
    Unified handler for all datasets:
    - If batch["image"] exists → use it (EuroSAT, BigEarthNet)
    - If DFC2020 → combine optical + sar to 15-channel input
    - Else raise error
    """
    # Case 1: standard datasets (EuroSAT...)
    if "image" in batch:
        return batch["image"].float().to(device)

    # Case 2: DFC2020 (optical=13ch, sar=2ch)
    if "optical" in batch:
        optical = batch["optical"].float().to(device)

        if "sar" in batch:
            sar = batch["sar"].float().to(device)
            return torch.cat([optical, sar], dim=1)  # 15 channels

        return optical  # some datasets may not have SAR

    raise KeyError("No image-like key found in batch: expected one of ['image', 'optical', 'sar']")


#############################################
# Metrics
#############################################

class Accuracy:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        preds = outputs.argmax(dim=1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.numel()

    def compute(self):
        return {"accuracy": self.correct / self.total if self.total > 0 else 0.0}


class SegmentationMetrics:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.total_loss = 0.0
        self.samples = 0
        self.class_names = class_names or [str(i) for i in range(num_classes)]

    def reset(self):
        self.conf_matrix[:] = 0
        self.total_loss = 0.0
        self.samples = 0

    def update(self, preds, targets, loss=0.0):
        preds_lbl = preds.argmax(dim=1).cpu().numpy()
        targets_np = targets.cpu().numpy()
        B = preds_lbl.shape[0]

        for b in range(B):
            pred_flat = preds_lbl[b].ravel()
            targ_flat = targets_np[b].ravel()
            for t, p in zip(targ_flat, pred_flat):
                if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                    self.conf_matrix[t, p] += 1

        self.total_loss += loss * B
        self.samples += B

    def compute(self):
        acc = np.diag(self.conf_matrix).sum() / (self.conf_matrix.sum() + 1e-8)
        iou = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            denom = (
                    self.conf_matrix[i, :].sum()
                    + self.conf_matrix[:, i].sum()
                    - self.conf_matrix[i, i]
            )
            iou[i] = self.conf_matrix[i, i] / (denom + 1e-8)

        mean_iou = np.nanmean(iou)
        mean_loss = self.total_loss / max(1, self.samples)

        return {"pixel_accuracy": acc, "mean_iou": mean_iou, "loss": mean_loss}


#############################################
# Trainer
#############################################

class Trainer:
    def __init__(
            self,
            model,
            criterion,
            optimizer,
            device="cuda",
            scheduler=None,
            mixed_precision=False,
            checkpoint_dir=Path("checkpoints"),
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.amp.GradScaler() if mixed_precision else None
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": []}
        self.best_metric = None

    ###############################
    # TRAIN EPOCH
    ###############################
    def train_epoch(self, loader, log_interval=50):
        self.model.train()
        running_loss = 0.0
        steps = 0

        for i, batch in enumerate(tqdm(loader, desc="Train")):
            imgs = _get_images_from_batch(batch, self.device)

            # get mask/labels
            if "mask" in batch:
                masks = batch["mask"]
            elif "label" in batch:
                masks = batch["label"]
            elif "labels" in batch:
                masks = batch["labels"]
            else:
                raise KeyError("No mask/label/labels found in batch")

            masks = masks.to(self.device)
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast(self.device):
                    preds = self.model(imgs)
                    loss = self.criterion(preds, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(imgs)
                loss = self.criterion(preds, masks)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            steps += 1

            if (i + 1) % log_interval == 0:
                avg = running_loss / steps
                print(f"[Train] iter {i + 1}, avg_loss={avg:.4f}")

        avg_loss = running_loss / max(1, steps)
        return {"loss": avg_loss}

    ###############################
    # VALIDATE
    ###############################
    @torch.no_grad()
    def validate(self, loader, metrics_obj=None):
        self.model.eval()
        running_loss = 0.0
        steps = 0

        if metrics_obj:
            metrics_obj.reset()
            is_seg = isinstance(metrics_obj, SegmentationMetrics)
        else:
            is_seg = False

        for batch in tqdm(loader, desc="Val"):
            imgs = _get_images_from_batch(batch, self.device)

            if "mask" in batch:
                masks = batch["mask"].to(self.device)
            elif "label" in batch:
                masks = batch["label"].to(self.device)
            elif "labels" in batch:
                masks = batch["labels"].to(self.device)
            else:
                raise KeyError("No mask/label/labels found in batch")

            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)

            preds = self.model(imgs)
            loss = self.criterion(preds, masks)

            running_loss += loss.item()
            steps += 1

            if metrics_obj:
                if is_seg:
                    metrics_obj.update(preds, masks, loss.item())
                else:
                    metrics_obj.update(preds, masks)

        avg_loss = running_loss / max(1, steps)
        results = {"loss": avg_loss}

        if metrics_obj:
            results.update(metrics_obj.compute())

        return results

    ###############################
    # FIT LOOP
    ###############################
    def fit(
            self,
            train_loader,
            val_loader,
            num_epochs,
            metrics_class,
            num_classes,
            log_interval=50,
            save_best_only=True,
            checkpoint_metric="loss",
    ):
        for epoch in range(1, num_epochs + 1):
            print(f"Epoch {epoch}/{num_epochs}")

            tr = self.train_epoch(train_loader, log_interval)
            val = self.validate(val_loader, metrics_class(num_classes))

            print(
                f"Epoch {epoch} - train_loss: {tr['loss']:.4f}, val_loss: {val['loss']:.4f}"
            )

            self.history["train_loss"].append(tr["loss"])
            self.history["val_loss"].append(val["loss"])

            if self.scheduler and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step()

            metric_val = val.get(checkpoint_metric, val["loss"])
            is_best = (self.best_metric is None) or (metric_val < self.best_metric)

            if is_best:
                self.best_metric = metric_val
                self._save_checkpoint(epoch, best=True)
            else:
                if not save_best_only:
                    self._save_checkpoint(epoch, best=False)

        return self.history

    def _save_checkpoint(self, epoch, best=False):
        path = self.checkpoint_dir / (
            f"best_model.pth" if best else f"epoch_{epoch}.pth"
        )
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, path)
        print(f"Saved checkpoint: {path}")
