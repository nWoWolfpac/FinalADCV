# training_segnet.py - SegNet training for DFC2020 segmentation
import torch
import argparse
from config import DEVICE, STAGE2, NUM_CLASSES
from src.data.dataset_utils import create_dfc2020_loaders
from src.models.encoder import EncoderClassifier
from src.models.segnet import SegNet
from src.utils import Trainer, SegmentationMetrics
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def visualize_predictions(model, loader, device, save_dir, max_samples=5):
    """Visualize predictions using 12-band input (RGB + NDVI + Radar)"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    count = 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].float().to(device)
            masks = batch["mask"].to(device)
            preds = model(imgs).argmax(dim=1).cpu()

            imgs = imgs.cpu()
            masks = masks.cpu()

            # Optical channels
            B02 = imgs[:, 2]  # Blue
            B03 = imgs[:, 3]  # Green
            B04 = imgs[:, 4]  # Red
            B08 = imgs[:, 8]  # NIR

            # Radar channels
            VV = imgs[:, 0]
            VH = imgs[:, 1]

            # NDVI
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
                radar_comp = np.stack([norm(VV[i]), norm(VH[i]), np.zeros_like(VV[i])], axis=-1)

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
                plt.savefig(f"{save_dir}/sample_{count}.png")
                plt.close()
                count += 1


def print_history_summary(history):
    print("\n========== TRAINING SUMMARY ==========")
    for epoch, data in enumerate(history):
        print(f"Epoch {epoch+1}: Train Loss = {data['train_loss']:.4f}, "
              f"Val Loss = {data['val_loss']:.4f}, "
              f"mIoU = {data['metrics']['mean_iou']:.4f}, "
              f"Acc = {data['metrics']['pixel_accuracy']:.4f}")
    print("======================================\n")


def save_history_csv(history, save_path="train_history.csv"):
    keys = ["epoch", "train_loss", "val_loss", "mean_iou", "pixel_accuracy"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for epoch, data in enumerate(history):
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": data["train_loss"],
                "val_loss": data["val_loss"],
                "mean_iou": data["metrics"]["mean_iou"],
                "pixel_accuracy": data["metrics"]["pixel_accuracy"]
            })
    print(f">>> Training history saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="SegNet training for DFC2020")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "resnet101"],
                        help="ResNet backbone for SegNet")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Optional path to pretrained encoder weights")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes in encoder classifier head")
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

    print(f">>> Loaded {args.backbone} encoder")

    # Load dataset
    train_loader, val_loader, test_loader = create_dfc2020_loaders(
        batch_size=STAGE2["batch_size"],
        input_size=STAGE2["input_size"],
        num_workers=STAGE2["num_workers"],
    )

    # Init SegNet
    model = SegNet(
        num_classes=NUM_CLASSES,
        backbone=args.backbone,
        encoder_weights_path=None,
        input_channels=12,
        input_size=STAGE2["input_size"]
    ).to(DEVICE)

    model.encoder = encoder_model.encoder
    print(f">>> Using SegNet with backbone {args.backbone}")

    # Loss + optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW([
        {"params": model.get_encoder_parameters(), "lr": STAGE2["encoder_lr"]},
        {"params": model.get_decoder_parameters(), "lr": STAGE2["decoder_lr"]},
    ], weight_decay=STAGE2["weight_decay"])

    trainer = Trainer(
        model, criterion, optimizer,
        device=DEVICE,
        mixed_precision=STAGE2["mixed_precision"],
        checkpoint_dir="checkpoints_stage2"
    )

    # Train
    history = trainer.fit(
        train_loader, val_loader,
        num_epochs=STAGE2["num_epochs"],
        metrics_class=SegmentationMetrics,
        num_classes=NUM_CLASSES,
        log_interval=STAGE2["log_interval"],
        save_best_only=STAGE2["save_best_only"],
        checkpoint_metric=STAGE2["checkpoint_metric"]
    )

    print(">>> Stage 2 training done!")
    print_history_summary(history)
    save_history_csv(history, "checkpoints_stage2/train_history.csv")

    # Load best model
    best_ckpt = "checkpoints_stage2/best_model.pth"
    checkpoint = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f">>> Loaded best checkpoint: {best_ckpt}")

    # Visualization
    save_dir = "checkpoints_stage2/visualization_final"
    visualize_predictions(model, val_loader, DEVICE, save_dir)
    print(f">>> Saved visualizations to: {save_dir}")


if __name__ == "__main__":
    main()
