# src/models/deeplabv3plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.encoder import EncoderClassifier


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]
        res = [b(x) for b in self.branches]
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode="bilinear", align_corners=False)
        res.append(gp)
        x = torch.cat(res, dim=1)
        return self.project(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=8, backbone="resnet50", encoder_weights_path=None,
                 input_channels=12, input_size=224):
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size

        # Load pretrained encoder
        self.encoder_model = EncoderClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=True
        )
        if encoder_weights_path:
            self.encoder_model.load_encoder_weights(encoder_weights_path)

        self.encoder = self.encoder_model.encoder

        # ---- Split low/high level tùy backbone ----
        children = list(self.encoder.children())
        if backbone.startswith("resnet"):
            # ResNet BigEarthNetv2 chỉ có 1 child
            resnet = children[0]
            self.low_level = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.act1,
                resnet.maxpool,
                resnet.layer1
            )
            self.high_level = nn.Sequential(
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
            )
            if input_channels != 3:
                old_conv = self.low_level[0]
                new_conv = nn.Conv2d(
                    input_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                self.low_level[0] = new_conv
                print(f">>> Updated conv1 to {input_channels} input channels")

        elif backbone in ["mobilevit", "mobilenetv4_hybrid"]:
            # Với MobileViT/MobileNetV4, split khoảng 1/3 low, 2/3 high
            split_idx = len(children) // 3
            self.low_level = nn.Sequential(*children[:split_idx])
            self.high_level = nn.Sequential(*children[split_idx:])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Infer số channels
        self._infer_channels()

        # Reduce low-level channels
        self.low_reduce = nn.Sequential(
            nn.Conv2d(self.low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # ASPP + decoder + classifier
        self.aspp = ASPP(self.high_level_channels, 256)
        self.decoder = DecoderBlock(256, 48, 256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def _infer_channels(self):
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            low = self.low_level(dummy)
            high = self.high_level(low)
            self.low_level_channels = low.shape[1]
            self.high_level_channels = high.shape[1]
        self.train()

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        low = self.low_level(x)
        high = self.high_level(low)
        aspp = self.aspp(high)
        low_r = self.low_reduce(low)
        dec = self.decoder(aspp, low_r)
        out = F.interpolate(dec, size=(H, W), mode="bilinear", align_corners=False)
        out = self.classifier(out)
        return out

    def get_encoder_parameters(self):
        return list(self.encoder.parameters())

    def get_decoder_parameters(self):
        exclude_prefixes = ("encoder", "low_level", "high_level")
        return [p for n, p in self.named_parameters() if not any(n.startswith(prefix) for prefix in exclude_prefixes)]

    def load_encoder_weights(self, weights_path, strict=False):
        self.encoder_model.load_encoder_weights(weights_path, strict=strict)
