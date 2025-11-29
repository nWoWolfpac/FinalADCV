# evaluate_unet.py
"""
Script để đánh giá UNet model trên tập test và visualize predictions.
Sau khi training xong, chạy script này để:
1. Load best_model từ checkpoint
2. Đánh giá trên tập test với các metrics: Accuracy, IoU, mIoU
3. Visualize predictions và lưu vào thư mục visualizations
"""
import torch
import argparse
from pathlib import Path
from config import DEVICE, NUM_CLASSES, STAGE2
from src.data.dataset_utils import create_dfc2020_loaders
from src.models.encoder import EncoderClassifier
from src.models.unet import UNet
from src.utils import evaluate, visualize_predictions


def load_model(checkpoint_path, backbone="resnet50", num_classes_encoder=19, dropout=0.1):
    """
    Load UNet model từ checkpoint.
    
    Args:
        checkpoint_path: Đường dẫn đến file checkpoint (.pth)
        backbone: Backbone encoder (resnet50, resnet101, ...)
        num_classes_encoder: Số classes của encoder pretrained (19 cho BigEarthNet)
        dropout: Dropout rate cho decoder
    
    Returns:
        model: UNet model đã được load weights
    """
    print(f"\n{'='*60}")
    print(f"Loading model from: {checkpoint_path}")
    print(f"{'='*60}")
    
    # Load encoder (cần để khởi tạo UNet đúng cách)
    encoder_model = EncoderClassifier(
        num_classes=num_classes_encoder,
        backbone=backbone,
        pretrained=True
    ).to(DEVICE)
    
    # Khởi tạo UNet
    model = UNet(
        num_classes=NUM_CLASSES,
        backbone=backbone,
        encoder_weights_path=None,  # Sẽ load từ checkpoint
        input_channels=12,
        bilinear=True,
        dropout=dropout
    ).to(DEVICE)
    
    # Thay thế encoder với pretrained encoder
    model.encoder_model = encoder_model
    encoder = encoder_model.encoder
    
    if backbone.startswith("resnet"):
        resnet = list(encoder.children())[0]
        
        # Thay thế các encoder stages
        model.maxpool = resnet.maxpool
        model.layer1 = resnet.layer1
        model.layer2 = resnet.layer2
        model.layer3 = resnet.layer3
        model.layer4 = resnet.layer4
        
        # Thay thế inc layer
        old_conv = model.inc[0]
        model.inc = torch.nn.Sequential(
            old_conv,
            resnet.bn1,
            resnet.act1
        )
    
    # Load weights từ checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f">>> Model loaded successfully!")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Backbone: {backbone}")
    print(f"  - Num classes: {NUM_CLASSES}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate UNet model on test set and visualize predictions"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_unet/best_model.pth",
        help="Path to model checkpoint (default: checkpoints_unet/best_model.pth)"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50", "resnet101"],
        help="Backbone encoder (default: resnet50)"
    )
    parser.add_argument(
        "--num_classes_encoder",
        type=int,
        default=19,
        help="Number of classes in encoder classifier (default: 19 for BigEarthNet)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for decoder (default: 0.1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (uses STAGE2 config if None)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for DataLoader (uses STAGE2 config if None)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to visualize (default: 10)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="visualizations/test_predictions",
        help="Directory to save visualization images (default: visualizations/test_predictions)"
    )
    
    args = parser.parse_args()
    
    # Kiểm tra checkpoint tồn tại
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train the model first or specify the correct checkpoint path."
        )
    
    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        num_classes_encoder=args.num_classes_encoder,
        dropout=args.dropout
    )
    
    # Load test dataset
    print(f"\n{'='*60}")
    print("Loading test dataset...")
    print(f"{'='*60}")
    
    batch_size = args.batch_size or STAGE2["batch_size"]
    num_workers = args.num_workers or STAGE2["num_workers"]
    
    _, _, test_loader = create_dfc2020_loaders(
        batch_size=batch_size,
        input_size=STAGE2["input_size"],
        num_workers=num_workers
    )
    
    print(f">>> Test dataset loaded!")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Input size: {STAGE2['input_size']}")
    print(f"  - Test samples: {len(test_loader.dataset)}")
    
    # Evaluation
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    total_cm, per_class_acc, per_class_iou, mean_iou = evaluate(
        model=model,
        val_loader=test_loader,
        num_classes=NUM_CLASSES,
        device=DEVICE
    )
    
    # Tính overall accuracy
    overall_accuracy = total_cm.diag().sum().float() / total_cm.sum().float()
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {overall_accuracy.item() * 100:.2f}%")
    print(f"{'='*60}")
    
    # Visualization
    print(f"\n{'='*60}")
    print(f"Visualizing predictions (max {args.max_samples} samples)...")
    print(f"{'='*60}")
    
    visualize_predictions(
        model=model,
        loader=test_loader,
        device=DEVICE,
        save_dir=args.save_dir,
        max_samples=args.max_samples
    )
    
    print(f">>> Visualizations saved to: {args.save_dir}")
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Backbone: {args.backbone}")
    print(f"Overall Accuracy: {overall_accuracy.item() * 100:.2f}%")
    print(f"Mean IoU: {mean_iou * 100:.2f}%")
    print(f"Visualizations: {args.save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

