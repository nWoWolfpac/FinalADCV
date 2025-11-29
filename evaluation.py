# evaluation.py
import torch
import numpy as np
from tqdm import tqdm
import os
import argparse

from src.data.dataset_utils import create_dfc2020_loaders
from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.segnet import SegNet
from src.utils import SegmentationMetrics
from config import DEVICE, STAGE2, NUM_CLASSES


# Tên 8 lớp DFC2020
DFC2020_CLASSES = [
    "Forest",
    "Shrubland",
    "Grassland",
    "Wetlands",
    "Croplands",
    "Urban/Built-up",
    "Barren",
    "Water"
]


def build_model(model_name, backbone, num_classes, input_channels=12, input_size=96):
    """Build segmentation model"""
    if model_name == "deeplabv3":
        model = DeepLabV3Plus(
            num_classes=num_classes,
            backbone=backbone,
            encoder_weights_path=None,
            input_channels=input_channels,
            input_size=input_size
        )
    elif model_name == "segnet":
        model = SegNet(
            num_classes=num_classes,
            backbone=backbone,
            encoder_weights_path=None,
            input_channels=input_channels,
            input_size=input_size
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return False
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        epoch = checkpoint.get("epoch", "unknown")
        print(f"[OK] Loaded checkpoint from epoch {epoch}: {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print(f"[OK] Loaded raw state_dict: {checkpoint_path}")
    
    return True


@torch.no_grad()
def evaluate(model, test_loader, device, num_classes):
    """Evaluate model on test set"""
    model.eval()
    metrics = SegmentationMetrics(num_classes, class_names=DFC2020_CLASSES)
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        imgs = batch["image"].float().to(device)
        masks = batch["mask"].to(device)
        
        if masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        
        # Skip batch nếu toàn ignore_index
        if (masks == 255).all():
            continue
        
        logits = model(imgs)
        metrics.update(logits, masks)
    
    return metrics


def print_results(metrics):
    """Print evaluation results"""
    results = metrics.compute()
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nPixel Accuracy: {results['pixel_accuracy']*100:.2f}%")
    print(f"Mean IoU: {results['mean_iou']*100:.2f}%")
    
    # Per-class Accuracy (Recall)
    print("\n" + "-"*40)
    print("Per-class Accuracy (Recall):")
    print("-"*40)
    
    acc_per_class = np.zeros(metrics.num_classes)
    for i in range(metrics.num_classes):
        # Accuracy per class = TP / (TP + FN) = diagonal / row sum
        row_sum = metrics.conf_matrix[i, :].sum()
        acc_per_class[i] = metrics.conf_matrix[i, i] / (row_sum + 1e-8)
        print(f"  {DFC2020_CLASSES[i]:20s}: {acc_per_class[i]*100:.2f}%")
    
    mean_class_acc = acc_per_class.mean()
    print(f"\n  Mean Class Accuracy: {mean_class_acc*100:.2f}%")
    
    # Per-class IoU
    print("\n" + "-"*40)
    print("Per-class IoU:")
    print("-"*40)
    
    iou_per_class = np.zeros(metrics.num_classes)
    for i in range(metrics.num_classes):
        denom = (
            metrics.conf_matrix[i, :].sum()
            + metrics.conf_matrix[:, i].sum()
            - metrics.conf_matrix[i, i]
        )
        iou_per_class[i] = metrics.conf_matrix[i, i] / (denom + 1e-8)
        print(f"  {DFC2020_CLASSES[i]:20s}: {iou_per_class[i]*100:.2f}%")
    
    print("\n" + "="*60)
    
    # Add to results
    results['mean_class_accuracy'] = mean_class_acc
    results['per_class_accuracy'] = acc_per_class
    results['per_class_iou'] = iou_per_class
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DFC2020 segmentation model")
    parser.add_argument("--model", type=str, default="deeplabv3",
                        choices=["deeplabv3", "segnet"],
                        help="Model architecture")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "resnet101"],
                        help="Backbone network")
    parser.add_argument("--checkpoint", type=str, 
                        default="checkpoints_stage2/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--num_classes", type=int, default=8,
                        help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Backbone: {args.backbone}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    # Build model
    print("Building model...")
    model = build_model(
        model_name=args.model,
        backbone=args.backbone,
        num_classes=args.num_classes,
        input_channels=12,
        input_size=STAGE2["input_size"]
    )
    
    # Load checkpoint
    print("Loading checkpoint...")
    if not load_checkpoint(model, args.checkpoint, DEVICE):
        print("Failed to load checkpoint. Exiting.")
        return
    
    model.to(DEVICE)
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = create_dfc2020_loaders(
        batch_size=args.batch_size,
        input_size=STAGE2["input_size"],
        num_workers=STAGE2["num_workers"]
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    print("\nStarting evaluation...")
    metrics = evaluate(model, test_loader, DEVICE, args.num_classes)
    
    # Print results
    results = print_results(metrics)
    
    return results


if __name__ == "__main__":
    main()
