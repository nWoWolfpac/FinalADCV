"""
Debug script để kiểm tra nguyên nhân NaN loss
"""
import torch
import numpy as np
from config import DEVICE, STAGE2, NUM_CLASSES
from src.data.dataset_utils import create_dfc2020_loaders
from src.models.unet import UNet
from src.models.encoder import EncoderClassifier

def check_data_quality(loader, num_batches=5):
    """Kiểm tra chất lượng dữ liệu"""
    print("=" * 60)
    print("KIỂM TRA CHẤT LƯỢNG DỮ LIỆU")
    print("=" * 60)
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
            
        imgs = batch["image"].float()
        masks = batch["mask"]
        
        print(f"\nBatch {i+1}:")
        print(f"  Image shape: {imgs.shape}")
        print(f"  Mask shape: {masks.shape}")
        
        # Check images
        img_nan = torch.isnan(imgs).any().item()
        img_inf = torch.isinf(imgs).any().item()
        img_min = imgs.min().item()
        img_max = imgs.max().item()
        img_mean = imgs.mean().item()
        img_std = imgs.std().item()
        
        print(f"  Image stats:")
        print(f"    Min: {img_min:.4f}, Max: {img_max:.4f}")
        print(f"    Mean: {img_mean:.4f}, Std: {img_std:.4f}")
        print(f"    Has NaN: {img_nan}, Has Inf: {img_inf}")
        
        if img_nan or img_inf:
            print(f"    ⚠️  WARNING: Image contains NaN or Inf!")
        
        # Check masks
        mask_min = masks.min().item()
        mask_max = masks.max().item()
        mask_unique = torch.unique(masks).tolist()
        
        print(f"  Mask stats:")
        print(f"    Min: {mask_min}, Max: {mask_max}")
        print(f"    Unique values: {mask_unique}")
        print(f"    Valid pixels (0-7): {(masks >= 0) & (masks < NUM_CLASSES)}")
        print(f"    Ignored pixels (255): {(masks == 255).sum().item()}")
        
        if mask_max >= NUM_CLASSES and mask_max != 255:
            print(f"    ⚠️  WARNING: Mask has invalid class values!")
    
    print("\n" + "=" * 60)


def check_model_output(model, loader, num_batches=3):
    """Kiểm tra output của model"""
    print("=" * 60)
    print("KIỂM TRA MODEL OUTPUT")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
                
            imgs = batch["image"].float().to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            
            # Fix mask shape: squeeze if needed (same as in training)
            if masks.dim() == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            
            try:
                preds = model(imgs)
                
                print(f"\nBatch {i+1}:")
                print(f"  Input shape: {imgs.shape}")
                print(f"  Output shape: {preds.shape}")
                print(f"  Mask shape: {masks.shape}")
                
                # Check predictions
                pred_nan = torch.isnan(preds).any().item()
                pred_inf = torch.isinf(preds).any().item()
                pred_min = preds.min().item()
                pred_max = preds.max().item()
                pred_mean = preds.mean().item()
                
                print(f"  Prediction stats:")
                print(f"    Min: {pred_min:.4f}, Max: {pred_max:.4f}")
                print(f"    Mean: {pred_mean:.4f}")
                print(f"    Has NaN: {pred_nan}, Has Inf: {pred_inf}")
                
                if pred_nan or pred_inf:
                    print(f"    ⚠️  WARNING: Predictions contain NaN or Inf!")
                
                # Check if predictions are too large (can cause NaN in softmax)
                if abs(pred_max) > 100 or abs(pred_min) > 100:
                    print(f"    ⚠️  WARNING: Predictions have very large values (>{100}), this can cause NaN in loss!")
                
                # Check loss
                criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
                loss = criterion(preds, masks)
                
                print(f"  Loss: {loss.item():.4f}")
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"    ⚠️  WARNING: Loss is NaN or Inf!")
                    
                    # Debug loss
                    print(f"    Debugging loss...")
                    print(f"    Mask unique: {torch.unique(masks).tolist()}")
                    print(f"    Valid pixels: {((masks >= 0) & (masks < NUM_CLASSES)).sum().item()}")
                    print(f"    Ignored pixels: {(masks == 255).sum().item()}")
                    
                    # Check if all pixels are ignored
                    valid_mask = (masks >= 0) & (masks < NUM_CLASSES)
                    if valid_mask.sum() == 0:
                        print(f"    ⚠️  ERROR: All pixels are ignored! This causes NaN loss.")
                
            except Exception as e:
                print(f"  ⚠️  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)


def check_normalization():
    """Kiểm tra normalization values"""
    print("=" * 60)
    print("KIỂM TRA NORMALIZATION")
    print("=" * 60)
    
    from config import SENTINEL1_MEAN, SENTINEL1_STD, SENTINEL2_MEAN, SENTINEL2_STD
    
    print(f"SENTINEL1_MEAN: {SENTINEL1_MEAN}")
    print(f"SENTINEL1_STD: {SENTINEL1_STD}")
    print(f"SENTINEL2_MEAN: {SENTINEL2_MEAN[:5]}... (showing first 5)")
    print(f"SENTINEL2_STD: {SENTINEL2_STD[:5]}... (showing first 5)")
    
    # Check for zero std
    if any(s == 0 for s in SENTINEL1_STD):
        print("⚠️  WARNING: SENTINEL1_STD contains zeros!")
    if any(s == 0 for s in SENTINEL2_STD):
        print("⚠️  WARNING: SENTINEL2_STD contains zeros!")
    
    print("\n" + "=" * 60)


def main():
    print("DEBUGGING NaN LOSS ISSUE")
    print("=" * 60)
    
    # 1. Check normalization
    check_normalization()
    
    # 2. Load dataset
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = create_dfc2020_loaders(
        batch_size=4,  # Small batch for debugging
        input_size=STAGE2["input_size"],
        num_workers=0,  # Set to 0 for debugging
    )
    
    # 3. Check data quality
    check_data_quality(train_loader, num_batches=5)
    
    # 4. Load model
    print("\nLoading model...")
    encoder_model = EncoderClassifier(
        num_classes=19,
        backbone="resnet50",
        pretrained=True
    ).to(DEVICE)
    
    model = UNet(
        num_classes=NUM_CLASSES,
        backbone="resnet50",
        encoder_weights_path=None,
        input_channels=12,
        bilinear=True,
        dropout=0.1
    ).to(DEVICE)
    
    # Replace encoder
    model.encoder_model = encoder_model
    encoder = encoder_model.encoder
    
    if True:  # resnet50
        resnet = list(encoder.children())[0]
        model.maxpool = resnet.maxpool
        model.layer1 = resnet.layer1
        model.layer2 = resnet.layer2
        model.layer3 = resnet.layer3
        model.layer4 = resnet.layer4
        
        import torch.nn as nn
        old_conv = model.inc[0]
        model.inc = nn.Sequential(
            old_conv,
            resnet.bn1,
            resnet.act1
        )
    
    print("Model loaded successfully!")
    
    # 5. Check model output
    check_model_output(model, train_loader, num_batches=3)
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

