# training_unetpp.py
"""
Training script for UNet++ segmentation model
Supports multiple backbones and optional deep supervision
"""

import torch
import argparse
from pathlib import Path
from config import DEVICE, STAGE2, NUM_CLASSES
from src.data.dataset_utils import create_dfc2020_loaders
from src.models.encoder import EncoderClassifier
from src.models.unetplusplus import UNetPlusPlus
from src.utils import Trainer, SegmentationMetrics
import matplotlib.pyplot as plt
import numpy as np
import csv


def visualize_predictions(model, loader, device, save_dir, max_samples=5):
    """Visualize predictions with RGB, NDVI, Radar, GT and Prediction"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    count = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].float().to(device)  # (B,12,H,W)
            masks = batch["mask"].to(device)
            
            # Handle deep supervision output
            outputs = model(imgs)
            if isinstance(outputs, list):
                preds = outputs[0].argmax(dim=1).cpu()  # Use deepest output
            else:
                preds = outputs.argmax(dim=1).cpu()

            imgs = imgs.cpu()
            masks = masks.cpu()

            # Extract bands
            # Sentinel-2 optical: bands 0-9
            B02 = imgs[:, 1]  # Blue
            B03 = imgs[:, 2]  # Green
            B04 = imgs[:, 3]  # Red
            B08 = imgs[:, 7]  # NIR
            
            # Sentinel-1 SAR: bands 10-11
            VV = imgs[:, 10]
            VH = imgs[:, 11]

            # Compute NDVI
            ndvi = (B08 - B04) / (B08 + B04 + 1e-6)

            def norm(x):
                x = x.numpy()
                return (x - x.min()) / (x.max() - x.min() + 1e-6)

            B = imgs.size(0)
            for i in range(B):
                if count >= max_samples:
                    return

                fig, ax = plt.subplots(1, 5, figsize=(20, 4))

                # RGB composite
                rgb = np.stack([
                    norm(B04[i]),
                    norm(B03[i]),
                    norm(B02[i])
                ], axis=-1)

                # Radar composite
                radar_comp = np.stack([
                    norm(VV[i]),
                    norm(VH[i]),
                    np.zeros_like(VV[i])
                ], axis=-1)

                # Plot
                ax[0].imshow(rgb)
                ax[0].set_title("RGB Composite")
                ax[0].axis("off")

                ax[1].imshow(ndvi[i].numpy(), cmap="RdYlGn")
                ax[1].set_title("NDVI")
                ax[1].axis("off")

                ax[2].imshow(radar_comp)
                ax[2].set_title("Radar (VV,VH)")
                ax[2].axis("off")

                # Squeeze mask if it has extra dimension (1, H, W) -> (H, W)
                mask_np = masks[i].squeeze().numpy()
                pred_np = preds[i].squeeze().numpy()
                
                ax[3].imshow(mask_np, cmap="tab10", vmin=0, vmax=7)
                ax[3].set_title("Ground Truth")
                ax[3].axis("off")

                ax[4].imshow(pred_np, cmap="tab10", vmin=0, vmax=7)
                ax[4].set_title("Prediction")
                ax[4].axis("off")

                plt.tight_layout()
                plt.savefig(save_dir / f"sample_{count}.png", dpi=150, bbox_inches='tight')
                plt.close()

                count += 1
                print(f"Saved visualization {count}/{max_samples}")


def print_history_summary(history):
    """Print training history summary"""
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for epoch, data in enumerate(history):
        msg = (
            f"Epoch {epoch+1:3d}: "
            f"Train Loss = {data['train_loss']:.4f}, "
            f"Val Loss = {data['val_loss']:.4f}, "
            f"mIoU = {data['metrics']['mean_iou']:.4f}, "
            f"Pixel Acc = {data['metrics']['pixel_accuracy']:.4f}"
        )
        print(msg)
    print("="*70 + "\n")


def save_history_csv(history, save_path):
    """Save training history to CSV"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    keys = ["epoch", "train_loss", "val_loss", "mean_iou", "pixel_accuracy"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for epoch, data in enumerate(history):
            row = {
                "epoch": epoch + 1,
                "train_loss": data["train_loss"],
                "val_loss": data["val_loss"],
                "mean_iou": data["metrics"]["mean_iou"],
                "pixel_accuracy": data["metrics"]["pixel_accuracy"]
            }
            writer.writerow(row)
    print(f">>> Training history saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train UNet++ for satellite image segmentation"
    )
    
    # Model arguments
    parser.add_argument(
        "--backbone", 
        type=str, 
        default="resnet50",
        choices=["resnet18", "resnet50", "resnet101", "mobilevit"],
        help="Backbone architecture for encoder (resnet18, resnet50, resnet101, mobilevit)"
    )
    parser.add_argument(
        "--deep_supervision", 
        action="store_true",
        help="Enable deep supervision (recommended for better training)"
    )
    
    # Encoder arguments
    parser.add_argument(
        "--encoder_checkpoint", 
        type=str, 
        default=None,
        help="Path to pretrained encoder weights from Stage-1"
    )
    parser.add_argument(
        "--num_classes_encoder", 
        type=int, 
        default=19,
        help="Number of classes in encoder (19 for BigEarthNet)"
    )
    
    # Training arguments
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (auto-generated if not specified)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization after training"
    )
    
    args = parser.parse_args()

    print("\n" + "="*70)
    print("UNet++ TRAINING CONFIGURATION")
    print("="*70)
    print(f"Backbone:          {args.backbone}")
    print(f"Deep Supervision:  {args.deep_supervision}")
    print(f"Device:            {DEVICE}")
    print(f"Batch Size:        {STAGE2['batch_size']}")
    print(f"Input Size:        {STAGE2['input_size']}")
    print(f"Num Epochs:        {STAGE2['num_epochs']}")
    print(f"Encoder LR:        {STAGE2['encoder_lr']}")
    print(f"Decoder LR:        {STAGE2['decoder_lr']}")
    print("="*70 + "\n")

    # ==================== LOAD PRETRAINED ENCODER ====================
    print(">>> Loading pretrained encoder...")
    encoder_model = EncoderClassifier(
        num_classes=args.num_classes_encoder,
        backbone=args.backbone,
        pretrained=True
    ).to(DEVICE)

    if args.encoder_checkpoint:
        encoder_model.load_encoder_weights(args.encoder_checkpoint)
        print(f">>> Loaded encoder weights from: {args.encoder_checkpoint}")
    else:
        print(f">>> Using HuggingFace pretrained {args.backbone} encoder")

    # ==================== LOAD DATASET ====================
    print("\n>>> Loading DFC2020 dataset...")
    train_loader, val_loader, test_loader = create_dfc2020_loaders(
        batch_size=STAGE2["batch_size"],
        input_size=STAGE2["input_size"],
        num_workers=STAGE2["num_workers"],
    )
    print(f">>> Train batches: {len(train_loader)}")
    print(f">>> Val batches:   {len(val_loader)}")
    print(f">>> Test batches:  {len(test_loader)}")

    # ==================== INITIALIZE UNET++ ====================
    print("\n>>> Initializing UNet++ model...")
    model = UNetPlusPlus(
        num_classes=NUM_CLASSES,
        backbone=args.backbone,
        encoder_weights_path=None,
        input_channels=12,
        input_size=STAGE2["input_size"],
        deep_supervision=args.deep_supervision
    ).to(DEVICE)

    # Replace encoder with pretrained one
    model.encoder = encoder_model.encoder
    print(f">>> Replaced encoder with pretrained {args.backbone}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.get_encoder_parameters())
    decoder_params = sum(p.numel() for p in model.get_decoder_parameters())
    print(f">>> Total parameters:   {total_params:,}")
    print(f">>> Encoder parameters: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f">>> Decoder parameters: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")

    # ==================== LOSS & OPTIMIZER ====================
    if args.deep_supervision:
        base_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        
        def deep_supervision_loss(outputs, targets):
            """Weighted loss for deep supervision"""
            weights = [1.0, 0.8, 0.6, 0.4]  # Deepest to shallowest
            total_loss = 0.0
            for out, w in zip(outputs, weights):
                total_loss += w * base_criterion(out, targets)
            return total_loss / sum(weights)
        
        criterion = deep_supervision_loss
        print("\n>>> Using deep supervision loss (weights: [1.0, 0.8, 0.6, 0.4])")
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        print("\n>>> Using standard CrossEntropyLoss")

    optimizer = torch.optim.AdamW([
        {"params": model.get_encoder_parameters(), "lr": STAGE2["encoder_lr"]},
        {"params": model.get_decoder_parameters(), "lr": STAGE2["decoder_lr"]},
    ], weight_decay=STAGE2["weight_decay"])

    # ==================== CHECKPOINT DIRECTORY ====================
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        ds_suffix = "_ds" if args.deep_supervision else ""
        checkpoint_dir = Path(f"checkpoints_unetpp_{args.backbone}{ds_suffix}")
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> Checkpoints will be saved to: {checkpoint_dir}")

    # ==================== TRAINER ====================
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        device=DEVICE,
        mixed_precision=STAGE2["mixed_precision"],
        checkpoint_dir=str(checkpoint_dir)
    )

    # ==================== TRAINING ====================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=STAGE2["num_epochs"],
        metrics_class=SegmentationMetrics,
        num_classes=NUM_CLASSES,
        log_interval=STAGE2["log_interval"],
        save_best_only=STAGE2["save_best_only"],
        checkpoint_metric=STAGE2["checkpoint_metric"],
    )

    print("\n>>> Training completed!")

    # ==================== PRINT HISTORY SUMMARY ====================
    print_history_summary(history)
    # Note: train_history.csv already saved by Trainer to checkpoint_dir

    # ==================== LOAD BEST MODEL ====================
    best_ckpt = checkpoint_dir / "best_model.pth"
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n>>> Loaded best checkpoint from: {best_ckpt}")
    else:
        print(f"\n>>> Warning: Best checkpoint not found at {best_ckpt}")

    # ==================== VISUALIZATION ====================
    if args.visualize:
        print("\n>>> Generating visualizations...")
        vis_dir = checkpoint_dir / "visualizations"
        visualize_predictions(model, val_loader, DEVICE, vis_dir, max_samples=10)
        print(f">>> Visualizations saved to: {vis_dir}")

    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model:            UNet++ ({args.backbone})")
    print(f"Deep Supervision: {args.deep_supervision}")
    print(f"Checkpoints:      {checkpoint_dir}")
    print(f"Best Model:       {best_ckpt}")
    print(f"History CSV:      {checkpoint_dir / 'train_history.csv'}")
    if args.visualize:
        print(f"Visualizations:   {checkpoint_dir / 'visualizations'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
