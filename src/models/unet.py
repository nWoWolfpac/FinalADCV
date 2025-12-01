# src/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder import EncoderClassifier


class DoubleConv(nn.Module):
    """
    (Conv => BN => ReLU) * 2 - Standard U-Net block
    Improved with better initialization and optional dropout
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        
        self.double_conv = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """
    Upsampling block with skip connection (U-Net decoder)
    Improved with better size handling and bilinear interpolation option
    
    Args:
        x1_channels: Number of channels in x1 (from decoder, lower resolution)
        x2_channels: Number of channels in x2 (from encoder skip connection)
        out_channels: Number of output channels after processing
    """
    
    def __init__(self, x1_channels, x2_channels, out_channels, bilinear=True, dropout=0.0):
        super().__init__()
        
        # Use bilinear upsampling or transposed conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Reduce channels of x1 before concatenation
            # x1_channels -> x1_channels // 2
            self.reduce = nn.Conv2d(x1_channels, x1_channels // 2, kernel_size=1, bias=False)
            # After concatenation: (x1_channels // 2 + x2_channels) -> out_channels
            self.conv = DoubleConv(x1_channels // 2 + x2_channels, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(x1_channels, x1_channels // 2, kernel_size=2, stride=2)
            self.reduce = None
            # After concatenation: (x1_channels // 2 + x2_channels) -> out_channels
            self.conv = DoubleConv(x1_channels // 2 + x2_channels, out_channels, dropout=dropout)
    
    def forward(self, x1, x2):
        """
        x1: from decoder (lower resolution) - shape: (B, x1_channels, H1, W1)
        x2: from encoder skip connection (higher resolution) - shape: (B, x2_channels, H2, W2)
        """
        x1 = self.up(x1)
        
        # Handle size mismatch (if input not perfectly divisible)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        if diffX > 0 or diffY > 0:
            x1 = F.pad(x1, [
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2
            ])
        
        # Reduce channels if using bilinear
        if self.reduce is not None:
            x1 = self.reduce(x1)
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation with pretrained encoder from BigEarthNet.
    Based on DFC2020 baseline repository patterns.
    
    Architecture:
        Encoder: ResNet50/101 pretrained on BigEarthNet (12 channels: 2 radar + 10 optical)
        Decoder: U-Net style with improved skip connections at each level
        Output: Segmentation masks (8 classes for DFC2020)
    
    Input: 12 channels (2 Sentinel-1 radar + 10 Sentinel-2 optical)
    """
    
    def __init__(
        self, 
        num_classes=8, 
        backbone="resnet50", 
        encoder_weights_path=None,
        input_channels=12,
        bilinear=True,
        dropout=0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.bilinear = bilinear
        
        # Fixed to 12 channels (S1 + S2)
        if input_channels != 12:
            raise ValueError(f"UNet only supports 12 input channels (got {input_channels})")
        
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
            # Initial convolution
            self.inc = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.act1
            )
            
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            
            # Update first conv to accept multi-channel input
            if input_channels != 3:
                old_conv = resnet.conv1
                old_in_channels = old_conv.in_channels
                
                new_conv = nn.Conv2d(
                    input_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                
                # Initialize new conv weights
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                
                # Copy pretrained weights if compatible
                with torch.no_grad():
                    if input_channels == old_in_channels:
                        # Same number of channels, copy directly
                        new_conv.weight.copy_(old_conv.weight)
                    elif input_channels > old_in_channels and old_in_channels > 0:
                        # More channels in new conv, copy what we can
                        # Copy first old_in_channels from old_conv to first old_in_channels of new_conv
                        new_conv.weight[:, :old_in_channels] = old_conv.weight[:, :old_in_channels]
                        # Initialize remaining channels
                        if input_channels > old_in_channels:
                            nn.init.kaiming_normal_(new_conv.weight[:, old_in_channels:], mode='fan_out', nonlinearity='relu')
                    # If input_channels < old_in_channels, just use initialized weights
                
                self.inc = nn.Sequential(
                    new_conv,
                    resnet.bn1,
                    resnet.act1
                )
                print(f">>> Updated conv1 to accept {input_channels} input channels (was {old_in_channels}, S1+S2)")
            
            # Dynamically infer channel sizes from encoder layers
            self._infer_channels(input_size=64)
            
            # ===== DECODER with U-Net SKIP CONNECTIONS =====
            # Each decoder block upsamples and concatenates with encoder skip
            # Args: (x1_channels, x2_channels, out_channels)
            # x1: from decoder (lower resolution), x2: from encoder skip (higher resolution)
            # Use dynamically inferred channels
            self.up1 = Up(self.channels[4], self.channels[3], self.channels[3], bilinear, dropout=dropout)
            self.up2 = Up(self.channels[3], self.channels[2], self.channels[2], bilinear, dropout=dropout)
            self.up3 = Up(self.channels[2], self.channels[1], self.channels[1], bilinear, dropout=dropout)
            self.up4 = Up(self.channels[1], self.channels[0], self.channels[1] // 2, bilinear, dropout=dropout)
            
            # Final upsampling to original resolution
            final_channels = self.channels[1] // 2
            if bilinear:
                self.up5 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    DoubleConv(final_channels, final_channels // 2, dropout=dropout)
                )
            else:
                self.up5 = nn.Sequential(
                    nn.ConvTranspose2d(final_channels, final_channels // 2, kernel_size=2, stride=2),
                    nn.BatchNorm2d(final_channels // 2),
                    nn.ReLU(inplace=True),
                    DoubleConv(final_channels // 2, final_channels // 2, dropout=dropout)
                )
            
            # Output layer
            self.outc = nn.Conv2d(final_channels // 2, num_classes, kernel_size=1)
            
            # Initialize output layer
            nn.init.xavier_uniform_(self.outc.weight)
            nn.init.constant_(self.outc.bias, 0)
            
            print(f">>> UNet decoder initialized with channels: inc={self.channels[0]}, layer1={self.channels[1]}, "
                  f"layer2={self.channels[2]}, layer3={self.channels[3]}, layer4={self.channels[4]}")
            
        elif backbone in ["mobilevit", "mobilenetv4_hybrid"]:
            raise NotImplementedError(f"U-Net with {backbone} backbone not yet implemented. Use ResNet.")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def _infer_channels(self, input_size=64):
        """
        Dynamically infer channel sizes from encoder layers by forward pass.
        This allows the model to work with different ResNet architectures (18, 50, 101).
        """
        # Get device from model parameters if available, otherwise use CPU
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # Model has no parameters yet (shouldn't happen, but safe fallback)
            device = torch.device('cpu')
        
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 12, input_size, input_size, device=device)
            x1 = self.inc(dummy)
            x2 = self.maxpool(x1)
            x2 = self.layer1(x2)
            x3 = self.layer2(x2)
            x4 = self.layer3(x3)
            x5 = self.layer4(x4)
            
            # Store channel sizes: [inc, layer1, layer2, layer3, layer4]
            self.channels = [
                x1.shape[1],  # inc output channels
                x2.shape[1],  # layer1 output channels
                x3.shape[1],  # layer2 output channels
                x4.shape[1],  # layer3 output channels
                x5.shape[1],  # layer4 output channels
            ]
        self.train()
    
    def forward(self, x):
        """
        Forward pass with skip connections at multiple scales
        
        Args:
            x: Input tensor (B, 12, H, W) - 12 channels (S1+S2)
        
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # ===== ENCODER with SKIP CONNECTIONS =====
        # Channel sizes are dynamically inferred based on backbone
        x1 = self.inc(x)              # (B, C0, H/2, W/2) - Skip 0
        x2 = self.maxpool(x1)         # (B, C0, H/4, W/4)
        x2 = self.layer1(x2)          # (B, C1, H/4, W/4) - Skip 1
        x3 = self.layer2(x2)          # (B, C2, H/8, W/8) - Skip 2
        x4 = self.layer3(x3)          # (B, C3, H/16, W/16) - Skip 3
        x5 = self.layer4(x4)          # (B, C4, H/32, W/32) - Bottleneck
        
        # ===== DECODER with U-Net SKIP CONNECTIONS =====
        d4 = self.up1(x5, x4)  # Combine layer4 (C4) + layer3 (C3) -> C3
        d3 = self.up2(d4, x3)   # Combine up1 (C3) + layer2 (C2) -> C2
        d2 = self.up3(d3, x2)   # Combine up2 (C2) + layer1 (C1) -> C1
        d1 = self.up4(d2, x1)   # Combine up3 (C1) + inc (C0) -> C1//2
        
        # Final upsampling to original resolution
        out = self.up5(d1)      # (B, C1//4, H, W)
        logits = self.outc(out)  # (B, num_classes, H, W)
        
        return logits
    
    def get_encoder_parameters(self):
        """Get parameters of pretrained encoder for differential learning rates"""
        params = []
        params.extend(list(self.inc.parameters()))
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
    
    def get_model_info(self):
        """Get model information"""
        encoder_params = sum(p.numel() for p in self.get_encoder_parameters())
        decoder_params = sum(p.numel() for p in self.get_decoder_parameters())
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'backbone': self.backbone,
            'input_channels': 12,
            'num_classes': self.num_classes,
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'total_params': total_params,
            'encoder_ratio': encoder_params / total_params * 100
        }


if __name__ == "__main__":
    # Test U-Net architecture
    print("=" * 60)
    print("Testing U-Net Architecture")
    print("=" * 60)
    
    # Test with 12 channels (S1+S2)
    print("\nTesting UNet with 12 channels (S1+S2):")
    model = UNet(
        num_classes=8, 
        backbone="resnet50", 
        input_channels=12
    )
    
    dummy_input = torch.randn(2, 12, 96, 96)
    output = model(dummy_input)
    
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    info = model.get_model_info()
    print(f"\n  Model Info:")
    print(f"    Input channels: {info['input_channels']}")
    print(f"    Encoder parameters: {info['encoder_params']:,}")
    print(f"    Decoder parameters: {info['decoder_params']:,}")
    print(f"    Total parameters:   {info['total_params']:,}")
    print(f"    Encoder ratio:      {info['encoder_ratio']:.1f}%")
    
    print("\n" + "=" * 60)
    print("U-Net architecture test passed! ✓")
    print("=" * 60)
