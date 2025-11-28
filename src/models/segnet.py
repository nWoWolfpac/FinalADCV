import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import EncoderClassifier


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class SegNet(nn.Module):
    """Simple SegNet-like decoder on top of pretrained encoder backbone.

    - Reuses EncoderClassifier to get an encoder (same backbones as DeepLabV3Plus).
    - Replaces first conv to accept arbitrary input_channels (e.g. 12 for DFC2020).
    - Builds a lightweight upsampling decoder to output segmentation logits.

    This is intentionally simple and generic to share as much code as possible
    with the existing DeepLabV3Plus pipeline (Trainer, dataset, optimizer setup).
    """

    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = "resnet50",
        encoder_weights_path: str | None = None,
        input_channels: int = 12,
        input_size: int = 224,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        self.backbone = backbone

        # Encoder from the same EncoderClassifier used elsewhere
        self.encoder_model = EncoderClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=True,
        )
        if encoder_weights_path:
            self.encoder_model.load_encoder_weights(encoder_weights_path)

        self.encoder = self.encoder_model.encoder

        children = list(self.encoder.children())
        if backbone.startswith("resnet"):
            # For BigEarthNetv2 ResNet-based encoder (single child)
            resnet = children[0]
            self.stem = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.act1,
                resnet.maxpool,
            )
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

            # Adapt first conv to multi-channel input
            if input_channels != 3:
                old_conv = self.stem[0]
                new_conv = nn.Conv2d(
                    input_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None,
                )
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                self.stem[0] = new_conv
        else:
            # For other backbones you can extend this mapping similarly
            raise ValueError(f"SegNet currently supports only ResNet-like backbones, got: {backbone}")

        # Infer channel dimensions by a dummy forward
        self._infer_channels()

        # Simple upsampling decoder: 4 stages
        self.up4 = UpBlock(self.c4, self.c3)
        self.up3 = UpBlock(self.c3, self.c2)
        self.up2 = UpBlock(self.c2, self.c1)
        self.up1 = UpBlock(self.c1, self.c0)

        self.classifier = nn.Conv2d(self.c0, num_classes, kernel_size=1)

    def _infer_channels(self) -> None:
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            x0 = self.stem(x)          # (B, c0, H/4, W/4)
            x1 = self.layer1(x0)       # (B, c1, ...)
            x2 = self.layer2(x1)       # (B, c2, ...)
            x3 = self.layer3(x2)       # (B, c3, ...)
            x4 = self.layer4(x3)       # (B, c4, ...)

            self.c0 = x0.shape[1]
            self.c1 = x1.shape[1]
            self.c2 = x2.shape[1]
            self.c3 = x3.shape[1]
            self.c4 = x4.shape[1]
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # Encoder
        x0 = self.stem(x)      # low-level
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Decoder (no skips here to keep it minimal; can be extended if needed)
        d4 = self.up4(x4)
        d3 = self.up3(d4)
        d2 = self.up2(d3)
        d1 = self.up1(d2)

        out = F.interpolate(d1, size=(H, W), mode="bilinear", align_corners=False)
        out = self.classifier(out)
        return out

    # Parameter groups for optimizer, mirroring DeepLabV3Plus API
    def get_encoder_parameters(self):
        return list(self.encoder.parameters())

    def get_decoder_parameters(self):
        encoder_prefixes = ("encoder", "stem", "layer1", "layer2", "layer3", "layer4")
        return [
            p
            for n, p in self.named_parameters()
            if not any(n.startswith(pref) for pref in encoder_prefixes)
        ]

    def load_encoder_weights(self, weights_path: str, strict: bool = False) -> None:
        self.encoder_model.load_encoder_weights(weights_path, strict=strict)
