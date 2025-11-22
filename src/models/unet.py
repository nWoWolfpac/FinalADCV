# src/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder import EncoderClassifier


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2 - Standard U-Net block"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upsampling block with skip connection (U-Net decoder)"""
    
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        
        # Use bilinear upsampling or transposed conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        x1: from decoder (lower resolution)
        x2: from encoder skip connection (higher resolution)
        """
        x1 = self.up(x1)
        
        # Handle size mismatch (if input not perfectly divisible)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation with pretrained encoder from BigEarthNet.
    
    Architecture:
        Encoder: ResNet50 pretrained on BigEarthNet (19 classes, 12 channels)
        Decoder: U-Net style with skip connections at each level
        Output: Segmentation masks (8 classes for DFC2020)
    """
    
    def __init__(self, num_classes=8, backbone="resnet50", encoder_weights_path=None,
                 input_channels=12, bilinear=False):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.bilinear = bilinear
        
        # Load pretrained encoder from BigEarthNet
        self.encoder_model = EncoderClassifier(
            num_classes=19,  # BigEarthNet has 19 classes
            backbone=backbone,
            pretrained=True
        )
        
        # Load encoder weights if provided
        if encoder_weights_path:
            self.encoder_model.load_encoder_weights(encoder_weights_path)
            print(f">>> Loaded encoder weights from {encoder_weights_path}")
        
        # Extract encoder and split into stages
        encoder = self.encoder_model.encoder
        
        if backbone.startswith("resnet"):
            resnet = list(encoder.children())[0]
            
            # ===== ENCODER STAGES with SKIP CONNECTIONS =====
            self.conv1 = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.act1
            )  # Output: (B, 64, H/2, W/2)
            
            self.maxpool = resnet.maxpool  # Output: (B, 64, H/4, W/4)
            self.layer1 = resnet.layer1    # Output: (B, 256, H/4, W/4)
            self.layer2 = resnet.layer2    # Output: (B, 512, H/8, W/8)
            self.layer3 = resnet.layer3    # Output: (B, 1024, H/16, W/16)
            self.layer4 = resnet.layer4    # Output: (B, 2048, H/32, W/32) - bottleneck
            
            # Update first conv to accept 12 channels instead of 3
            if input_channels != 3:
                old_conv = resnet.conv1
                new_conv = nn.Conv2d(
                    input_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                self.conv1 = nn.Sequential(
                    new_conv,
                    resnet.bn1,
                    resnet.act1
                )
                print(f">>> Updated conv1 to accept {input_channels} input channels")
            
            # ===== DECODER with U-Net SKIP CONNECTIONS =====
            # Each decoder block upsamples and concatenates with encoder skip
            self.up1 = Up(2048 + 1024, 1024, bilinear)  # Combine layer4 + layer3
            self.up2 = Up(1024 + 512, 512, bilinear)    # Combine up1 + layer2
            self.up3 = Up(512 + 256, 256, bilinear)     # Combine up2 + layer1
            self.up4 = Up(256 + 64, 128, bilinear)      # Combine up3 + conv1
            
            # Final upsampling to original resolution
            self.up5 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                DoubleConv(64, 64)
            )
            
            # Output layer
            self.outc = nn.Conv2d(64, num_classes, kernel_size=1)
            
        elif backbone in ["mobilevit", "mobilenetv4_hybrid"]:
            raise NotImplementedError(f"U-Net with {backbone} backbone not yet implemented. Use ResNet.")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        """
        Forward pass with skip connections at multiple scales
        
        Args:
            x: Input tensor (B, 12, H, W)
        
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # ===== ENCODER with SKIP CONNECTIONS =====
        x1 = self.conv1(x)              # (B, 64, H/2, W/2)
        x2 = self.maxpool(x1)           # (B, 64, H/4, W/4)
        x2 = self.layer1(x2)            # (B, 256, H/4, W/4)
        x3 = self.layer2(x2)            # (B, 512, H/8, W/8)
        x4 = self.layer3(x3)            # (B, 1024, H/16, W/16)
        x5 = self.layer4(x4)            # (B, 2048, H/32, W/32) - bottleneck
        
        # ===== DECODER with U-Net SKIP CONNECTIONS =====
        d4 = self.up1(x5, x4)  # (B, 1024, H/16, W/16) - combine with layer3
        d3 = self.up2(d4, x3)  # (B, 512, H/8, W/8)    - combine with layer2
        d2 = self.up3(d3, x2)  # (B, 256, H/4, W/4)    - combine with layer1
        d1 = self.up4(d2, x1)  # (B, 128, H/2, W/2)    - combine with conv1
        
        # Final upsampling to original resolution
        out = self.up5(d1)     # (B, 64, H, W)
        logits = self.outc(out)  # (B, num_classes, H, W)
        
        return logits
    
    def get_encoder_parameters(self):
        """Get parameters of pretrained encoder for differential learning rates"""
        params = []
        params.extend(list(self.conv1.parameters()))
        params.extend(list(self.layer1.parameters()))
        params.extend(list(self.layer2.parameters()))
        params.extend(list(self.layer3.parameters()))
        params.extend(list(self.layer4.parameters()))
        return params
    
    def get_decoder_parameters(self):
        """Get parameters of decoder for differential learning rates"""
        params = []
        params.extend(list(self.up1.parameters()))
        params.extend(list(self.up2.parameters()))
        params.extend(list(self.up3.parameters()))
        params.extend(list(self.up4.parameters()))
        params.extend(list(self.up5.parameters()))
        params.extend(list(self.outc.parameters()))
        return params
    
    def freeze_encoder(self):
        """Freeze encoder weights for initial training"""
        for param in self.get_encoder_parameters():
            param.requires_grad = False
        print(">>> Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.get_encoder_parameters():
            param.requires_grad = True
        print(">>> Encoder unfrozen")


if __name__ == "__main__":
    # Test U-Net architecture
    print("=" * 60)
    print("Testing U-Net Architecture")
    print("=" * 60)
    
    model = UNet(num_classes=8, backbone="resnet50", input_channels=12)
    
    # Test with dummy input
    dummy_input = torch.randn(2, 12, 96, 96)
    output = model(dummy_input)
    
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    encoder_params = sum(p.numel() for p in model.get_encoder_parameters())
    decoder_params = sum(p.numel() for p in model.get_decoder_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nEncoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters:   {total_params:,}")
    print(f"Encoder ratio:      {encoder_params / total_params * 100:.1f}%")
    
    print("\n" + "=" * 60)
    print("U-Net architecture test passed! ✓")
    print("=" * 60)

