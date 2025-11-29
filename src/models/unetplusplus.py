# src/models/unetplusplus.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder import EncoderClassifier

#!


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class NestedDecoderBlock(nn.Module):
    """Nested decoder block for UNet++ with multiple skip connections"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        total_in_channels = sum(in_channels_list)
        self.conv = ConvBlock(total_in_channels, out_channels)

    def forward(self, *inputs):
        # Upsample first input to match size of skip connections
        x_up = F.interpolate(inputs[0], size=inputs[1].shape[2:], 
                            mode="bilinear", align_corners=False)
        # Concatenate upsampled feature with all skip connections
        x = torch.cat([x_up] + list(inputs[1:]), dim=1)
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    UNet++ implementation with pretrained encoder support
    
    Architecture follows nested U-Net structure with dense skip connections
    """
    def __init__(self, num_classes=8, backbone="resnet50", encoder_weights_path=None,
                 input_channels=12, input_size=224, deep_supervision=False):
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        self.deep_supervision = deep_supervision
        self.backbone_name = backbone

        # Load pretrained encoder
        self.encoder_model = EncoderClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=True
        )
        if encoder_weights_path:
            self.encoder_model.load_encoder_weights(encoder_weights_path)

        self.encoder = self.encoder_model.encoder

        # Build encoder stages based on backbone
        self._build_encoder_stages()

        # Infer channels for each encoder stage
        self._infer_encoder_channels()

        # Build nested decoder
        self._build_nested_decoder()

        # Final classifiers
        if deep_supervision:
            # Each output has different number of channels from its decoder block
            self.final_conv_0_4 = nn.Conv2d(self.decoder_channels[0], num_classes, kernel_size=1)  # x0_4 from dec0_4
            self.final_conv_0_3 = nn.Conv2d(self.decoder_channels[3], num_classes, kernel_size=1)  # x0_3 from dec0_3
            self.final_conv_0_2 = nn.Conv2d(self.decoder_channels[2], num_classes, kernel_size=1)  # x0_2 from dec0_2
            self.final_conv_0_1 = nn.Conv2d(self.decoder_channels[1], num_classes, kernel_size=1)  # x0_1 from dec0_1
        else:
            self.final_conv = nn.Conv2d(self.decoder_channels[0], num_classes, kernel_size=1)

    def _build_encoder_stages(self):
        """Split encoder into stages based on backbone type"""
        children = list(self.encoder.children())
        
        if self.backbone_name.startswith("resnet"):
            resnet = children[0]
            
            # Stage 0: Initial conv + bn + relu
            self.enc0 = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.act1
            )
            
            # Stage 1-4: ResNet layers
            self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)
            self.enc2 = resnet.layer2
            self.enc3 = resnet.layer3
            self.enc4 = resnet.layer4
            
            # Update first conv for multi-channel input
            if self.input_channels != 3:
                old_conv = self.enc0[0]
                new_conv = nn.Conv2d(
                    self.input_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                self.enc0[0] = new_conv
                print(f">>> Updated ResNet conv1 to {self.input_channels} input channels")

        elif self.backbone_name == "mobilevit":
            m = self.encoder.vision_encoder
            
            self.enc0 = m.stem
            self.enc1 = m.stages[0]
            self.enc2 = m.stages[1]
            self.enc3 = m.stages[2]
            self.enc4 = m.stages[3]
            
            if self.input_channels != 3:
                old_conv = m.stem.conv
                new_conv = nn.Conv2d(
                    self.input_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                m.stem.conv = new_conv
                print(f">>> Updated MobileViT first conv to {self.input_channels} channels")

        else:
            raise ValueError(
                f"Unsupported backbone: {self.backbone_name}. "
                f"Supported backbones: resnet18, resnet50, resnet101, mobilevit"
            )

    def _infer_encoder_channels(self):
        """Infer output channels for each encoder stage"""
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            
            x0 = self.enc0(dummy)
            x1 = self.enc1(x0)
            x2 = self.enc2(x1)
            x3 = self.enc3(x2)
            x4 = self.enc4(x3)
            
            self.enc_channels = [
                x0.shape[1],  # C0
                x1.shape[1],  # C1
                x2.shape[1],  # C2
                x3.shape[1],  # C3
                x4.shape[1]   # C4
            ]
            
            print(f">>> Encoder channels: {self.enc_channels}")
        self.train()

    def _build_nested_decoder(self):
        """Build nested decoder with dense skip connections"""
        # Decoder channels (typically half of encoder channels)
        self.decoder_channels = [
            max(32, self.enc_channels[0] // 2),
            max(64, self.enc_channels[1] // 2),
            max(128, self.enc_channels[2] // 2),
            max(256, self.enc_channels[3] // 2)
        ]
        
        # Nested decoder blocks
        # X^0_1: upsample from X^0_2 + skip from X^1_0
        self.dec0_1 = NestedDecoderBlock(
            [self.enc_channels[2], self.enc_channels[1]], 
            self.decoder_channels[1]
        )
        
        # X^0_2: upsample from X^0_3 + skip from X^1_0 and X^0_1
        self.dec0_2 = NestedDecoderBlock(
            [self.enc_channels[3], self.enc_channels[1], self.decoder_channels[1]], 
            self.decoder_channels[2]
        )
        
        # X^0_3: upsample from X^0_4 + skip from X^1_0, X^0_1, X^0_2
        self.dec0_3 = NestedDecoderBlock(
            [self.enc_channels[4], self.enc_channels[1], self.decoder_channels[1], self.decoder_channels[2]], 
            self.decoder_channels[3]
        )
        
        # X^0_4: final upsample + all previous skips
        self.dec0_4 = NestedDecoderBlock(
            [self.decoder_channels[3], self.enc_channels[0]], 
            self.decoder_channels[0]
        )
        
        # Additional nested blocks for level 1
        self.dec1_1 = NestedDecoderBlock(
            [self.enc_channels[3], self.enc_channels[2]], 
            self.decoder_channels[1]
        )
        
        self.dec1_2 = NestedDecoderBlock(
            [self.enc_channels[4], self.enc_channels[2], self.decoder_channels[1]], 
            self.decoder_channels[2]
        )
        
        # Additional nested blocks for level 2
        self.dec2_1 = NestedDecoderBlock(
            [self.enc_channels[4], self.enc_channels[3]], 
            self.decoder_channels[2]
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        
        # Encoder
        x0 = self.enc0(x)   # 1/2
        x1 = self.enc1(x0)  # 1/4
        x2 = self.enc2(x1)  # 1/8
        x3 = self.enc3(x2)  # 1/16
        x4 = self.enc4(x3)  # 1/32
        
        # Nested decoder
        # Level 1
        x1_1 = self.dec1_1(x3, x2)
        x0_1 = self.dec0_1(x2, x1)
        
        # Level 2
        x2_1 = self.dec2_1(x4, x3)
        x1_2 = self.dec1_2(x4, x2, x1_1)
        x0_2 = self.dec0_2(x3, x1, x0_1)
        
        # Level 3
        x0_3 = self.dec0_3(x4, x1, x0_1, x0_2)
        
        # Level 4 (final)
        x0_4 = self.dec0_4(x0_3, x0)
        
        # Upsample to original size
        x0_4 = F.interpolate(x0_4, size=(H, W), mode="bilinear", align_corners=False)
        
        if self.deep_supervision:
            # Return multiple outputs for deep supervision
            x0_1 = F.interpolate(x0_1, size=(H, W), mode="bilinear", align_corners=False)
            x0_2 = F.interpolate(x0_2, size=(H, W), mode="bilinear", align_corners=False)
            x0_3 = F.interpolate(x0_3, size=(H, W), mode="bilinear", align_corners=False)
            
            out1 = self.final_conv_0_1(x0_1)
            out2 = self.final_conv_0_2(x0_2)
            out3 = self.final_conv_0_3(x0_3)
            out4 = self.final_conv_0_4(x0_4)
            
            return [out4, out3, out2, out1]  # Return from deepest to shallowest
        else:
            return self.final_conv(x0_4)

    def get_encoder_parameters(self):
        """Get encoder parameters for differential learning rates"""
        return list(self.encoder.parameters())

    def get_decoder_parameters(self):
        """Get decoder parameters for differential learning rates"""
        exclude_prefixes = ("encoder", "enc0", "enc1", "enc2", "enc3", "enc4")
        return [p for n, p in self.named_parameters() 
                if not any(n.startswith(prefix) for prefix in exclude_prefixes)]

    def load_encoder_weights(self, weights_path, strict=False):
        """Load pretrained encoder weights"""
        self.encoder_model.load_encoder_weights(weights_path, strict=strict)
