# src/models/deeplabv3plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6,12,18]):
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
            nn.Conv2d(out_channels*(len(rates)+2), out_channels, kernel_size=1, bias=False),
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
    def __init__(self, encoder, backbone="resnet50", num_classes=8, input_channels=12, input_size=224):
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.input_channels = input_channels
        self.input_size = input_size

        # ---- Split low/high level tùy backbone ----
        if backbone.startswith("resnet"):
            # Nếu encoder không có conv1 chuẩn, tạo một conv đầu riêng
            if self.input_channels != 3 or not hasattr(self.encoder, "conv1"):
                self.input_conv = nn.Sequential(
                    nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
            else:
                self.input_conv = nn.Identity()  # pass through conv1 of encoder

            self.low_level = getattr(self.encoder, "layer1", nn.Identity())
            self.high_level = nn.Sequential(
                getattr(self.encoder, "layer2", nn.Identity()),
                getattr(self.encoder, "layer3", nn.Identity()),
                getattr(self.encoder, "layer4", nn.Identity())
            )

        elif backbone == "mobilevit":
            m = self.encoder
            # Kiểm tra input_channels
            expected_in = 3  # MobileViT mặc định dùng 3 channels
            if self.input_channels != expected_in:
                # Thêm conv mapping từ input_channels -> expected_in
                self.input_conv = nn.Conv2d(
                    in_channels=self.input_channels,
                    out_channels=expected_in,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            else:
                self.input_conv = nn.Identity()

            self.low_level = nn.Sequential(
                self.input_conv,
                m.stem,
                m.stages[0]
            )
            self.high_level = nn.Sequential(*m.stages[1:])


        elif backbone == "mobilenetv4_hybrid":
            m = self.encoder
            # Kiểm tra input_channels
            expected_in = 3
            if self.input_channels != expected_in:
                self.input_conv = nn.Conv2d(
                    in_channels=self.input_channels,
                    out_channels=expected_in,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            else:
                self.input_conv = nn.Identity()
            self.low_level = nn.Sequential(
                self.input_conv,
                m.conv_stem,
                m.bn1,
                m.blocks[0],
                m.blocks[1]
            )
            self.high_level = nn.Sequential(*m.blocks[2:])


        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # --- debug shapes ---
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            low_debug = self.input_conv(dummy)
            low_debug = self.low_level(low_debug)
            high_debug = self.high_level(low_debug)
        self.train()

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

    # ----------------- Utility methods -----------------
    def get_encoder_parameters(self):
        return list(self.encoder.parameters())

    def get_decoder_parameters(self):
        exclude_prefixes = ("encoder", "low_level", "high_level")
        return [p for n,p in self.named_parameters() if not any(n.startswith(prefix) for prefix in exclude_prefixes)]
