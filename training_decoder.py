# train_stage2.py
import torch
import argparse
from config import DEVICE, STAGE2, NUM_CLASSES
from src.data.dataset_utils import create_dfc2020_loaders
from src.models.encoder import EncoderBackbone
from src.models.deeplabv3plus import DeepLabV3Plus
from src.utils import Trainer, SegmentationMetrics
import csv


# ======================================================
# PRINT TRAINING HISTORY
# ======================================================
def print_history_summary(history):
    print("\n========== TRAINING SUMMARY ==========")
    for epoch, data in enumerate(history):
        msg = (
            f"Epoch {epoch + 1}: "
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
# BUILD MODEL (usable from other scripts)
# ======================================================
def build_model(backbone="resnet50", num_classes=8,
                stage1_checkpoint=None, input_channels=12, input_size=224, device="cuda"):
    # Load Stage1 encoder
    encoder_model = EncoderBackbone(backbone=backbone, input_channels=input_channels)
    if stage1_checkpoint:
        encoder_model.load_encoder(stage1_checkpoint)
    print(">>> Stage 1 encoder loaded.")

    # Build DeepLabV3Plus Stage2
    model = DeepLabV3Plus(
        encoder=encoder_model.encoder,  # trả về feature map 4D
        backbone=backbone,
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=input_size
    ).to(device)
    print(f">>> DeepLabV3Plus built with {input_channels}-band input.")
    return model


# ======================================================
# MAIN SCRIPT
# ======================================================
def main():
    parser = argparse.ArgumentParser(description="Stage-2 training with DeepLabV3+ and pretrained encoder")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet50", "resnet101", "mobilevit", "mobilenetv4_hybrid"],
                        help="Backbone to load pretrained encoder")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Optional path to pretrained encoder weights")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes in encoder classifier head")
    args = parser.parse_args()

    # Load dataset
    train_loader, val_loader, test_loader = create_dfc2020_loaders(
        batch_size=STAGE2["batch_size"],
        input_size=STAGE2["input_size"],
        num_workers=STAGE2["num_workers"],
    )

    # Build model
    model = build_model(backbone=args.backbone, num_classes=args.num_classes,
                        device=DEVICE)

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
        checkpoint_dir="checkpoints_stage2"
    )

    # Train stage 2
    history = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=STAGE2["num_epochs"],
        metrics_class=SegmentationMetrics,
        num_classes=args.num_classes,
        log_interval=STAGE2["log_interval"],
        save_best_only=STAGE2["save_best_only"],
        checkpoint_metric=STAGE2["checkpoint_metric"]
    )

    print(">>> Stage 2 training done!")

    # Print training summary
    print_history_summary(history)
    save_history_csv(history, "checkpoints_stage2/train_history.csv")

    # Load best model (optional, useful for evaluation)
    best_ckpt = "checkpoints_stage2/best_model.pth"
    checkpoint = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f">>> Loaded best checkpoint: {best_ckpt}")

    return model, val_loader, test_loader


if __name__ == "__main__":
    main()
