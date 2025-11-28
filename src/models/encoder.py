# src/models/encoder.py

import torch
import torch.nn as nn
from reben.reben_publication.BigEarthNetv2_0_ImageClassifier import (
    BigEarthNetv2_0_ImageClassifier
)

BACKBONE_MODEL_IDS = {
    "resnet18": "BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0",
    "resnet50": "BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0",
    "resnet101": "BIFOLD-BigEarthNetv2-0/resnet101-all-v0.2.0",
    "mobilevit": "BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.2.0",
    "mobilenetv4_hybrid": "BIFOLD-BigEarthNetv2-0/mobilenetv4_hybrid_medium-all-v0.2.0",
}


def extract_encoder(model, backbone: str):
    """
    Tách encoder từ pretrained BigEarthNet model.
    Loại bỏ classifier/linear cuối.
    """
    encoder = model.model  # tất cả backbone đều có .model

    if backbone.startswith("resnet"):
        if hasattr(encoder, "fc"):
            encoder.fc = nn.Identity()
    elif backbone in ["mobilevit", "mobilenetv4_hybrid"]:
        if hasattr(encoder, "classifier"):
            encoder.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return encoder, None  # Stage2 dùng feature map nên không cần in_features


class EncoderBackbone(nn.Module):
    """Encoder Stage2 segmentation DeepLabV3Plus"""

    def __init__(self, backbone="resnet50", input_channels=12):
        super().__init__()
        self.backbone = backbone
        self.input_channels = input_channels

        # Load pretrained BigEarthNet encoder
        base_model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
            BACKBONE_MODEL_IDS[backbone]
        )

        # Extract encoder (bỏ classifier)
        encoder_base, _ = extract_encoder(base_model, backbone)

        # Nếu input_channels != 3, thêm conv 1x1 mapping -> 3 channels
        if input_channels != 3:
            self.input_conv = nn.Conv2d(input_channels, 3, kernel_size=1)
        else:
            self.input_conv = nn.Identity()

        self.encoder = encoder_base

    def forward(self, x):
        """Trả về feature map 4D cho decoder segmentation"""
        x = self.input_conv(x)
        x = self.encoder(x)
        return x

    def save_encoder(self, path):
        torch.save({"encoder_state_dict": self.encoder.state_dict()}, path)

    def load_encoder(self, path, strict=False):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("encoder_state_dict", ckpt)
        self.encoder.load_state_dict(sd, strict=strict)
