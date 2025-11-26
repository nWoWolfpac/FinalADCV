# config.py
from pathlib import Path
import torch

# Datasets

DATASET_DFC2020 = "GFM-Bench/DFC2020"      # dùng cho Stage2 segmentation

# I/O
CHECKPOINTS_DIR = Path("checkpoints")
LOGS_DIR = Path("logs")
VISUALIZATIONS_DIR = Path("visualizations")
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Hardware / seed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
SEED = 42

# Stage1: Encoder pretrain on BigEarthNet
STAGE1 = {
    "input_size": 224,
    "batch_size": 4,
    "num_epochs": 1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 1,
    "pin_memory": False,
    "save_path": CHECKPOINTS_DIR / "stage1_encoder.pth"
}

# Stage2: Segmentation on DFC2020
STAGE2 = {
    "input_size": 96,               # DFC2020 patch size
    "batch_size": 16,
    "num_epochs": 50,
    "freeze_encoder_epochs": 5,
    "encoder_lr": 1e-4,  # Giảm từ 5e-4 để tránh NaN
    "decoder_lr": 5e-5,  # Giảm từ 1e-4 để tránh NaN
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "min_lr": 1e-6,
    "num_workers": 4,
    "pin_memory": True,
    "mixed_precision": True,
    "gradient_accumulation_steps": 1,
    "save_best_only": True,
    "checkpoint_metric": "loss",
    "log_interval": 50,
    "encoder_checkpoint": STAGE1["save_path"]   # sử dụng pretrain từ Stage1
}

# Problem-specific: DFC2020
NUM_CLASSES = 8       # 8 lớp DFC2020
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

# Normalization placeholders (tính từ dữ liệu thật nếu cần)
SENTINEL1_MEAN = [0.0, 0.0]
SENTINEL1_STD = [1.0, 1.0]
SENTINEL2_MEAN = [0.0] * 13
SENTINEL2_STD = [1.0] * 13
USE_NORMALIZATION = True

