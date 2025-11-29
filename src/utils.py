# src/utils.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


#############################################
# NEW: unified loader for image data
#############################################
def _get_images_from_batch(batch, device):
    if isinstance(batch, dict):
        if "image" in batch:
            imgs = batch["image"]
        elif "optical" in batch:
            imgs = batch["optical"]
            if "sar" in batch:
                imgs = torch.cat([imgs, batch["sar"]], dim=1)
        else:
            raise KeyError("No image-like key found in batch: expected one of ['image','optical','sar']")
    elif isinstance(batch, list):
        imgs = torch.stack([b["image"] for b in batch], dim=0)
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(0)
    elif imgs.dim() != 4:
        raise ValueError(f"Expected 3D or 4D tensor for images, got {imgs.shape}")

    if torch.isnan(imgs).any() or torch.isinf(imgs).any():
        print("[DEBUG] Found NaN/inf in images batch")

    return imgs.float().to(device)


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
        mask = (targets >= 0) & (targets < self.num_classes)  # ignore 255
        self.correct += ((preds == targets) & mask).sum().item()
        self.total += mask.sum().item()

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
        preds = preds.argmax(dim=1)
        preds = preds.view(-1).cpu()
        targets = targets.view(-1).cpu()

        mask = (targets >= 0) & (targets < self.num_classes)  # ignore 255
        preds = preds[mask]
        targets = targets[mask]

        cm = torch.bincount(
            self.num_classes * targets + preds,
            minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

        self.conf_matrix += cm.numpy()
        self.total_loss += loss
        self.samples += 1

    def compute(self):
        acc = np.diag(self.conf_matrix).sum() / (self.conf_matrix.sum() + 1e-8)
        iou = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            denom = self.conf_matrix[i, :].sum() + self.conf_matrix[:, i].sum() - self.conf_matrix[i, i]
            iou[i] = self.conf_matrix[i, i] / (denom + 1e-8)
        mean_iou = np.nanmean(iou)
        mean_loss = self.total_loss / max(1, self.samples)
        return {"pixel_accuracy": acc, "mean_iou": mean_iou, "loss": mean_loss}


#############################################
# Trainer
#############################################
class Trainer:
    def __init__(self, model, criterion, optimizer, device="cuda",
                 scheduler=None, mixed_precision=False, checkpoint_dir=Path("checkpoints")):
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

    def train_epoch(self, loader, log_interval=50):
        self.model.train()
        running_loss = 0.0
        steps = 0

        for i, batch in enumerate(tqdm(loader, desc="Train")):
            imgs = _get_images_from_batch(batch, self.device)
            if "mask" in batch:
                masks = batch["mask"]
            elif "label" in batch:
                masks = batch["label"]
            elif "labels" in batch:
                masks = batch["labels"]
            else:
                raise KeyError("No mask/label/labels found in batch")
            masks = masks.to(self.device).long()
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            if (masks == 255).all():
                continue

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
                print(f"[Train] iter {i + 1}, avg_loss={running_loss / steps:.4f}")

        return {"loss": running_loss / max(1, steps)}

    @torch.no_grad()
    def validate(self, loader, metrics_obj=None):
        self.model.eval()
        running_loss = 0.0
        steps = 0
        is_seg = isinstance(metrics_obj, SegmentationMetrics) if metrics_obj else False

        for i, batch in enumerate(tqdm(loader, desc="Val")):
            imgs = _get_images_from_batch(batch, self.device)
            if "mask" in batch:
                masks = batch["mask"].to(self.device)
            elif "label" in batch:
                masks = batch["label"].to(self.device)
            elif "labels" in batch:
                masks = batch["labels"].to(self.device)
            else:
                raise KeyError("No mask/label/labels found in batch")
            masks = masks.long()
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            if (masks == 255).all():
                continue

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

    def fit(self, train_loader, val_loader, num_epochs, metrics_class, num_classes,
            log_interval=50, save_best_only=True, checkpoint_metric="loss",
            save_history=True, history_csv_path="train_history.csv",
            history_pickle_path="train_history.pkl"):

        history = []
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            # train
            tr = self.train_epoch(train_loader, log_interval)
            # validation
            val_metrics_obj = metrics_class(num_classes)
            val = self.validate(val_loader, val_metrics_obj)
            print(f"Epoch {epoch} - train_loss: {tr['loss']:.4f}, val_loss: {val['loss']:.4f}")

            # lưu history
            epoch_dict = {
                "train_loss": tr["loss"],
                "val_loss": val["loss"],
                "metrics": {k: v for k, v in val.items() if k != "loss"}
            }
            history.append(epoch_dict)

            # step LR scheduler nếu có
            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            # tính metric checkpoint
            metric_val = val.get(checkpoint_metric, val["loss"])
            is_best = (self.best_metric is None) or (metric_val < self.best_metric)
            if is_best:
                self.best_metric = metric_val

            # luôn lưu checkpoint mỗi epoch, best_model.pth update khi tốt nhất
            self._save_checkpoint(epoch, best=is_best)
            print(f"[DEBUG] Epoch {epoch} - checkpoint_metric={metric_val:.4f}, best_metric={self.best_metric:.4f}")

        # lưu history CSV / pickle
        if save_history:
            import csv, pickle
            keys = ["epoch", "train_loss", "val_loss"] + list(history[0]["metrics"].keys())
            with open(history_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for i, e in enumerate(history):
                    row = {"epoch": i + 1, "train_loss": e["train_loss"], "val_loss": e["val_loss"]}
                    row.update(e["metrics"])
                    writer.writerow(row)
            with open(history_pickle_path, "wb") as f:
                pickle.dump(history, f)
            print(f">>> Training history saved to {history_csv_path} and {history_pickle_path}")

        return history

    def _save_checkpoint(self, epoch, best=False):
        path = self.checkpoint_dir / (f"best_model.pth" if best else f"epoch_{epoch}.pth")
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(state, path)



#############################################
# POST-TRAINING EVALUATION / VISUALIZATION
#############################################
DFC2020_CLASSES_8 = [
    "Forest", "Shrubland", "Grassland", "Wetlands",
    "Croplands", "Urban/Built-up", "Barren", "Water"
]


def confusion_matrix(pred, gt, num_classes):
    pred = pred.view(-1)
    gt = gt.view(-1)
    mask = (gt >= 0) & (gt < num_classes)  # ignore 255
    pred = pred[mask]
    gt = gt[mask]
    cm = torch.bincount(num_classes * gt + pred, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def compute_iou_from_cm(cm):
    num_classes = cm.shape[0]
    IoU = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fn = cm[c, :].sum().item() - tp
        fp = cm[:, c].sum().item() - tp
        denom = tp + fp + fn
        IoU.append(tp / denom if denom > 0 else 0)
    mIoU = sum(IoU) / num_classes
    return IoU, mIoU


def per_class_accuracy_from_cm(cm):
    acc = []
    for c in range(cm.shape[0]):
        total = cm[c, :].sum().item()
        acc.append(cm[c, c].item() / total if total > 0 else 0)
    return acc


def evaluate(model, val_loader, num_classes=8, device="cuda"):
    model.eval()
    model.to(device)
    total_cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", ncols=100):
            imgs = _get_images_from_batch(batch, device)
            labels = batch["mask"].to(device).long()
            preds = model(imgs).argmax(dim=1)
            cm = confusion_matrix(preds, labels, num_classes)
            total_cm += cm.to(total_cm.device)

    iou, miou = compute_iou_from_cm(total_cm)
    acc = per_class_accuracy_from_cm(total_cm)

    print("\n==================== PER-CLASS ACCURACY ====================")
    for name, a in zip(DFC2020_CLASSES_8, acc):
        print(f"{name:15s}: {a * 100:.2f}%")
    print("\n==================== PER-CLASS IoU ====================")
    for name, i in zip(DFC2020_CLASSES_8, iou):
        print(f"{name:15s}: {i * 100:.2f}%")
    print(f"\n==================== mIoU: {miou * 100:.2f}% ====================")
    print("\n==================== CONFUSION MATRIX (8×8) ====================")
    print(total_cm)

    return total_cm, acc, iou, miou
