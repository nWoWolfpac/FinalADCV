# train_stage2.py
import torch
import argparse
from config import DEVICE, STAGE2, NUM_CLASSES
from src.data.dataset_utils import create_dfc2020_loaders
from src.models.encoder import EncoderClassifier
from src.models.deeplabv3plus import DeepLabV3Plus
from src.models.unetplusplus import UNetPlusPlus
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
            
            # Handle both single output and deep supervision (list of outputs)
            outputs = model(imgs)
            if isinstance(outputs, list):
                # Deep supervision: use the deepest (first) output
                preds = outputs[0].argmax(dim=1).cpu()
            else:
                preds = outputs.argmax(dim=1).cpu()

            imgs = imgs.cpu()
            masks = masks.cpu()

            # Sentinel-2 optical channels (10 bands)
            B02 = imgs[:, 1]  # Blue
            B03 = imgs[:, 2]  # Green
            B04 = imgs[:, 3]  # Red
            B08 = imgs[:, 7]  # NIR

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
    parser = argparse.ArgumentParser(description="Stage-2 training with segmentation models and pretrained encoder")
    parser.add_argument("--model", type=str, default="deeplabv3plus",
                        choices=["deeplabv3plus", "unetplusplus"],
                        help="Segmentation model architecture")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "resnet101", "mobilevit", "mobilenetv4_hybrid"],
                        help="Backbone to load pretrained encoder")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Optional path to pretrained encoder weights")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes in encoder classifier head")
    parser.add_argument("--deep_supervision", action="store_true",
                        help="Enable deep supervision for UNet++ (ignored for DeepLabV3+)")
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

    # Init segmentation model based on choice
    if args.model == "deeplabv3plus":
        model = DeepLabV3Plus(
            num_classes=NUM_CLASSES,
            backbone=args.backbone,
            encoder_weights_path=None,
            input_channels=12,
            input_size=STAGE2["input_size"]
        ).to(DEVICE)
        print(f">>> Initialized DeepLabV3+ with {args.backbone} backbone")
    
    elif args.model == "unetplusplus":
        model = UNetPlusPlus(
            num_classes=NUM_CLASSES,
            backbone=args.backbone,
            encoder_weights_path=None,
            input_channels=12,
            input_size=STAGE2["input_size"],
            deep_supervision=args.deep_supervision
        ).to(DEVICE)
        ds_status = "enabled" if args.deep_supervision else "disabled"
        print(f">>> Initialized UNet++ with {args.backbone} backbone (deep supervision: {ds_status})")
    
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Replace encoder with pretrained one
    model.encoder = encoder_model.encoder
    print(f">>> Updated {args.model} encoder with 12-band input.")

    # Loss + optimizer
    # For UNet++ with deep supervision, we need special loss handling
    if args.model == "unetplusplus" and args.deep_supervision:
        # Deep supervision loss: weighted sum of losses at different depths
        base_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
        
        def deep_supervision_loss(outputs, targets):
            """
            outputs: list of [out4, out3, out2, out1] from deepest to shallowest
            targets: ground truth mask
            """
            weights = [1.0, 0.8, 0.6, 0.4]  # More weight on deeper outputs
            total_loss = 0.0
            for out, w in zip(outputs, weights):
                total_loss += w * base_criterion(out, targets)
            return total_loss / sum(weights)
        
        criterion = deep_supervision_loss
        print(">>> Using deep supervision loss for UNet++")
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    optimizer = torch.optim.AdamW([
        {"params": model.get_encoder_parameters(), "lr": STAGE2["encoder_lr"]},
        {"params": model.get_decoder_parameters(), "lr": STAGE2["decoder_lr"]},
    ], weight_decay=STAGE2["weight_decay"])

    # Create checkpoint directory with model name
    checkpoint_dir = f"checkpoints_stage2_{args.model}_{args.backbone}"
    
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        device=DEVICE,
        mixed_precision=STAGE2["mixed_precision"],
        checkpoint_dir=checkpoint_dir
    )
    
    print(f">>> Checkpoints will be saved to: {checkpoint_dir}")

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

    print(">>> Stage 2 training done!")

    # Print training summary
    print_history_summary(history)
    save_history_csv(history, f"{checkpoint_dir}/train_history.csv")

    # Load best model
    best_ckpt = f"{checkpoint_dir}/best_model.pth"
    checkpoint = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f">>> Loaded best checkpoint: {best_ckpt}")

    # Visualization
    save_dir = f"{checkpoint_dir}/visualization_final"
    visualize_predictions(model, val_loader, DEVICE, save_dir)
    print(f">>> Saved visualizations to: {save_dir}")


if __name__ == "__main__":
    main()
