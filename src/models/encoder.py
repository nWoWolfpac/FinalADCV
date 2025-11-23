# src/models/encoder.py
import torch
import torch.nn as nn
from reben.reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

BACKBONE_MODEL_IDS = {
    "resnet18": "BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0",
    "resnet50": "BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0",
    "resnet101": "BIFOLD-BigEarthNetv2-0/resnet101-all-v0.2.0",
    "mobilevit": "BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.2.0",
    "mobilenetv4_hybrid": "BIFOLD-BigEarthNetv2-0/mobilenetv4_hybrid_medium-all-v0.2.0",
}


def extract_encoder(model, backbone: str):
    """Trích encoder và in_features cho backbone"""
    if backbone.startswith("resnet"):
        encoder = model.model
        head = getattr(encoder, "fc", None)
        if head is None:
            linear_layers = [m for m in encoder.modules() if isinstance(m, nn.Linear)]
            head = linear_layers[-1]
        in_features = head.in_features
    elif backbone == "mobilevit":
        encoder = model.model
        head = getattr(encoder, "classifier", None)
        if head is None:
            linear_layers = [m for m in encoder.modules() if isinstance(m, nn.Linear)]
            head = linear_layers[-1]
        in_features = head.in_features
    elif backbone == "mobilenetv4_hybrid":
        encoder = model.model
        head = getattr(encoder, "classifier", None)
        if head is None:
            linear_layers = [m for m in encoder.modules() if isinstance(m, nn.Linear)]
            head = linear_layers[-1]
        in_features = head.in_features
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return encoder, in_features


class EncoderClassifier(nn.Module):
    """Stage-1 encoder + classifier"""

    def __init__(self, num_classes: int, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        model_id = BACKBONE_MODEL_IDS[backbone]
        self.base_model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_id)

        # Extract encoder only
        self.encoder, in_features = extract_encoder(self.base_model, backbone)

        # Classifier riêng (num_classes stage1 có thể = 19)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x, return_features=False):
        """
        return_features=True -> trả về feature map 4D cho DeepLabV3Plus
        return_features=False -> logits classifier
        """
        features = self.encoder(x)
        if return_features:
            return features
        if features.ndim == 4:
            features = self.pool(features).flatten(1)
        return self.classifier(features)

    def save_encoder_weights(self, path):
        torch.save({"encoder_state_dict": self.encoder.state_dict()}, path)

    def load_encoder_weights(self, path, strict=False):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("encoder_state_dict", ckpt)
        self.encoder.load_state_dict(sd, strict=strict)
