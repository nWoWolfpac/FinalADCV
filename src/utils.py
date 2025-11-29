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
    """
    Unified handler for all datasets:
    - If batch["image"] exists → use it (EuroSAT, BigEarthNet)
    - If DFC2020 → combine optical + sar to 12-band input
    - Else raise error
    """
    if "image" in batch:
        return batch["image"].float().to(device)

    if "optical" in batch:
        optical = batch["optical"].float().to(device)
        if "sar" in batch:
            sar = batch["sar"].float().to(device)
            return torch.cat([optical, sar], dim=1)
        return optical

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
                
                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[ERROR] NaN/Inf loss at iteration {i+1}")
                    print(f"  Loss: {loss.item()}")
                    print(f"  Input stats: min={imgs.min().item():.4f}, max={imgs.max().item():.4f}, mean={imgs.mean().item():.4f}, has_nan={torch.isnan(imgs).any().item()}")
                    print(f"  Pred stats: min={preds.min().item():.4f}, max={preds.max().item():.4f}, mean={preds.mean().item():.4f}, has_nan={torch.isnan(preds).any().item()}")
                    print(f"  Mask stats: min={masks.min().item()}, max={masks.max().item()}, unique={torch.unique(masks).tolist()}")
                    print(f"  Valid pixels: {((masks >= 0) & (masks < preds.size(1))).sum().item()}")
                    print(f"  Ignored pixels: {(masks == 255).sum().item()}")
                    # Skip this batch
                    continue
                
                self.scaler.scale(loss).backward()
                # Gradient clipping to prevent explosion
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(imgs)
                loss = self.criterion(preds, masks)
                
                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[ERROR] NaN/Inf loss at iteration {i+1}")
                    print(f"  Loss: {loss.item()}")
                    print(f"  Input stats: min={imgs.min().item():.4f}, max={imgs.max().item():.4f}, mean={imgs.mean().item():.4f}, has_nan={torch.isnan(imgs).any().item()}")
                    print(f"  Pred stats: min={preds.min().item():.4f}, max={preds.max().item():.4f}, mean={preds.mean().item():.4f}, has_nan={torch.isnan(preds).any().item()}")
                    print(f"  Mask stats: min={masks.min().item()}, max={masks.max().item()}, unique={torch.unique(masks).tolist()}")
                    print(f"  Valid pixels: {((masks >= 0) & (masks < preds.size(1))).sum().item()}")
                    print(f"  Ignored pixels: {(masks == 255).sum().item()}")
                    # Skip this batch
                    continue
                
                loss.backward()
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        is_seg = isinstance(metrics_obj, SegmentationMetrics) if metrics_obj else False

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
            save_history=True,
            history_csv_path="train_history.csv",
            history_pickle_path="train_history.pkl"
    ):
        history = []
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            tr = self.train_epoch(train_loader, log_interval)
            val_metrics_obj = metrics_class(num_classes)
            val = self.validate(val_loader, val_metrics_obj)
            print(f"Epoch {epoch} - train_loss: {tr['loss']:.4f}, val_loss: {val['loss']:.4f}")
            epoch_dict = {
                "train_loss": tr["loss"],
                "val_loss": val["loss"],
                "metrics": {k: v for k, v in val.items() if k != "loss"}
            }
            history.append(epoch_dict)

            if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            metric_val = val.get(checkpoint_metric, val["loss"])
            is_best = (self.best_metric is None) or (metric_val < self.best_metric)

            if is_best:
                self.best_metric = metric_val
                self._save_checkpoint(epoch, best=True)
            else:
                if not save_best_only:
                    self._save_checkpoint(epoch, best=False)

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
            print(f">>> History saved as CSV: {history_csv_path}")

            with open(history_pickle_path, "wb") as f:
                pickle.dump(history, f)
            print(f">>> History saved as Pickle: {history_pickle_path}")

        return history

    def _save_checkpoint(self, epoch, best=False):
        path = self.checkpoint_dir / (f"best_model.pth" if best else f"epoch_{epoch}.pth")
        state = {"epoch": epoch,
                 "model_state_dict": self.model.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict()}
        torch.save(state, path)
        print(f"Saved checkpoint: {path}")


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
    mask = (gt >= 0) & (gt < num_classes)
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
            
            if "mask" in batch:
                labels = batch["mask"].to(device)
            elif "label" in batch:
                labels = batch["label"].to(device)
            elif "labels" in batch:
                labels = batch["labels"].to(device)
            else:
                raise KeyError("No mask/label/labels found in batch")
            
            # Handle mask shape: [B, 1, H, W] -> [B, H, W]
            if labels.dim() == 4 and labels.size(1) == 1:
                labels = labels.squeeze(1)
            
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


def visualize_predictions(model, loader, device, save_dir, max_samples=5):
    """
    Visualize predictions on test/val dataset.
    Assumes 12-channel input: [VV, VH, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in loader:
            imgs = _get_images_from_batch(batch, device)
            
            if "mask" in batch:
                masks = batch["mask"].to(device)
            elif "label" in batch:
                masks = batch["label"].to(device)
            elif "labels" in batch:
                masks = batch["labels"].to(device)
            else:
                raise KeyError("No mask/label/labels found in batch")
            
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            
            preds = model(imgs).argmax(dim=1).cpu()
            imgs = imgs.cpu()
            masks = masks.cpu()

            # 12-channel input: [VV, VH, B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
            VV = imgs[:, 0]   # Radar VV
            VH = imgs[:, 1]   # Radar VH
            B02 = imgs[:, 2]  # Blue
            B03 = imgs[:, 3]  # Green
            B04 = imgs[:, 4]  # Red
            B08 = imgs[:, 8]  # NIR
            
            ndvi = (B08 - B04) / (B08 + B04 + 1e-6)

            def norm(x):
                x = x.numpy()
                return (x - x.min()) / (x.max() - x.min() + 1e-6)

            B = imgs.size(0)
            for i in range(B):
                if count >= max_samples:
                    return
                fig, ax = plt.subplots(1, 5, figsize=(20, 4))
                rgb = np.stack([norm(B04[i]), norm(B03[i]), norm(B02[i])], axis=-1)
                radar_comp = np.stack([norm(VV[i]), norm(VH[i]), np.zeros_like(norm(VV[i]))], axis=-1)
                ax[0].imshow(rgb)
                ax[0].set_title("RGB Composite")
                ax[0].axis("off")
                ax[1].imshow(ndvi[i].numpy(), cmap="RdYlGn")
                ax[1].set_title("NDVI")
                ax[1].axis("off")
                ax[2].imshow(radar_comp)
                ax[2].set_title("Radar (VV,VH)")
                ax[2].axis("off")
                ax[3].imshow(masks[i].numpy(), cmap="viridis")
                ax[3].set_title("GT Mask")
                ax[3].axis("off")
                ax[4].imshow(preds[i].numpy(), cmap="viridis")
                ax[4].set_title("Prediction")
                ax[4].axis("off")
                plt.tight_layout()
                plt.savefig(f"{save_dir}/sample_{count}.png", dpi=150, bbox_inches='tight')
                plt.close()
                count += 1
