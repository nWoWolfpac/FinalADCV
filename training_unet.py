# training_unet.py
import torch
import torch.nn as nn
import argparse
from config import DEVICE, STAGE2, NUM_CLASSES
from src.data.dataset_utils import create_dfc2020_loaders
from src.models.encoder import EncoderClassifier
from src.models.unet import UNet
from src.utils import Trainer, SegmentationMetrics
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


# ======================================================
# VISUALIZE USING 12-BAND (RGB + NDVI + RADAR)
# ======================================================
def visualize_predictions(model, loader, device, save_dir, max_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    count = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].float().to(device)  # (B,12,H,W)
            masks = batch["mask"].to(device)
            preds = model(imgs).argmax(dim=1).cpu()

            imgs = imgs.cpu()
            masks = masks.cpu()

            # Sentinel-2 optical channels (10 bands)
            # Channels: [0:VV, 1:VH, 2:B02, 3:B03, 4:B04, 5:B05, 6:B06, 7:B07, 8:B08, 9:B8A, 10:B11, 11:B12]
            B02 = imgs[:, 2]  # Blue
            B03 = imgs[:, 3]  # Green
            B04 = imgs[:, 4]  # Red
            B08 = imgs[:, 8]  # NIR

            # Radar channels (VV, VH)
            VV = imgs[:, 0]
            VH = imgs[:, 1]

            # Compute NDVI
            ndvi = (B08 - B04) / (B08 + B04 + 1e-6)

            # Normalize helper
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

                ax[3].imshow(masks[i].numpy(), cmap="tab10", vmin=0, vmax=NUM_CLASSES-1)
                ax[3].set_title("GT Mask")
                ax[3].axis("off")

                ax[4].imshow(preds[i].numpy(), cmap="tab10", vmin=0, vmax=NUM_CLASSES-1)
                ax[4].set_title("Prediction")
                ax[4].axis("off")

                plt.tight_layout()
                plt.savefig(f"{save_dir}/sample_{count}.png", dpi=150)
                plt.close()

                count += 1


# ======================================================
# PRINT TRAINING HISTORY
# ======================================================
def print_history_summary(history):
    print("\n========== TRAINING SUMMARY ==========")
    for epoch, data in enumerate(history):
        msg = (
            f"Epoch {epoch+1}: "
            f"Train Loss = {data['train_loss']:.4f}, "
            f"Val Loss = {data['val_loss']:.4f}, "
            f"mIoU = {data['metrics']['mean_iou']:.4f}, "
            f"Acc = {data['metrics']['pixel_accuracy']:.4f}"
        )
        print(msg)
    print("======================================\n")


def save_history_csv(history, save_path="train_history.csv"):
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


# ======================================================
# MAIN SCRIPT
# ======================================================
def main():
    parser = argparse.ArgumentParser(description="Stage-2 training with UNet and pretrained encoder")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "resnet101"],
                        help="Backbone to load pretrained encoder")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Optional path to pretrained encoder weights")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes in encoder classifier head")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for decoder")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=None,
                        help="Number of epochs to freeze encoder (uses STAGE2 config if None)")
    args = parser.parse_args()

    # Load encoder
    encoder_model = EncoderClassifier(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=True
    ).to(DEVICE)

    if args.encoder_checkpoint:
        encoder_model.load_encoder_weights(args.encoder_checkpoint)
        print(f">>> Loaded encoder weights from {args.encoder_checkpoint}")

    print(f">>> Loaded {args.backbone} encoder with {args.num_classes} classes")

    # Load dataset
    train_loader, val_loader, test_loader = create_dfc2020_loaders(
        batch_size=STAGE2["batch_size"],
        input_size=STAGE2["input_size"],
        num_workers=STAGE2["num_workers"],
    )

    # Init UNet - sẽ tự động tạo encoder trong __init__
    model = UNet(
        num_classes=NUM_CLASSES,
        backbone=args.backbone,
        encoder_weights_path=None,  # Sẽ thay thế encoder sau
        input_channels=12,
        bilinear=True,
        dropout=args.dropout
    ).to(DEVICE)

    # Thay thế encoder_model và các encoder layers với pretrained ones
    # Điều này đảm bảo model sử dụng encoder đã được load weights từ checkpoint
    model.encoder_model = encoder_model
    encoder = encoder_model.encoder
    
    if args.backbone.startswith("resnet"):
        resnet = list(encoder.children())[0]
        
        # Thay thế các encoder stages với pretrained weights
        model.maxpool = resnet.maxpool
        model.layer1 = resnet.layer1
        model.layer2 = resnet.layer2
        model.layer3 = resnet.layer3
        model.layer4 = resnet.layer4
        
        # Thay thế inc layer: giữ lại conv đã được tạo cho 12 channels,
        # nhưng sử dụng bn1 và act1 từ pretrained encoder
        old_conv = model.inc[0]  # Conv đã được tạo cho 12 channels
        model.inc = nn.Sequential(
            old_conv,      # Giữ lại conv cho 12 channels
            resnet.bn1,    # Sử dụng pretrained bn1
            resnet.act1    # Sử dụng pretrained act1
        )
    
    print(f">>> Updated UNet encoder with pretrained weights and 12-band input (S1+S2)")

    # Loss + optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    optimizer = torch.optim.AdamW([
        {"params": model.get_encoder_parameters(), "lr": STAGE2["encoder_lr"]},
        {"params": model.get_decoder_parameters(), "lr": STAGE2["decoder_lr"]},
    ], weight_decay=STAGE2["weight_decay"])

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        device=DEVICE,
        mixed_precision=STAGE2["mixed_precision"],
        checkpoint_dir="checkpoints_unet"
    )

    # Freeze encoder for initial epochs if specified
    freeze_epochs = args.freeze_encoder_epochs if args.freeze_encoder_epochs is not None else STAGE2.get("freeze_encoder_epochs", 0)
    if freeze_epochs > 0:
        model.freeze_encoder()
        print(f">>> Encoder will be frozen for {freeze_epochs} epochs")

    # Train stage 2
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=STAGE2["num_epochs"],
        metrics_class=SegmentationMetrics,
        num_classes=NUM_CLASSES,
        log_interval=STAGE2["log_interval"],
        save_best_only=STAGE2["save_best_only"],
        checkpoint_metric=STAGE2["checkpoint_metric"]
    )

    # Unfreeze encoder after freeze period
    if freeze_epochs > 0:
        model.unfreeze_encoder()
        print(f">>> Encoder unfrozen after {freeze_epochs} epochs")

    print(">>> Stage 2 training done!")

    # Print training summary
    print_history_summary(history)
    save_history_csv(history, "checkpoints_unet/train_history.csv")

    # Load best model
    best_ckpt = "checkpoints_unet/best_model.pth"
    checkpoint = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f">>> Loaded best checkpoint: {best_ckpt}")

    # Print model info
    info = model.get_model_info()
    print(f"\n>>> Model Information:")
    print(f"    Backbone: {info['backbone']}")
    print(f"    Input channels: {info['input_channels']}")
    print(f"    Total parameters: {info['total_params']:,}")
    print(f"    Encoder ratio: {info['encoder_ratio']:.1f}%")

    # Visualization
    save_dir = "checkpoints_unet/visualization_final"
    visualize_predictions(model, val_loader, DEVICE, save_dir)
    print(f">>> Saved visualizations to: {save_dir}")


if __name__ == "__main__":
    main()

