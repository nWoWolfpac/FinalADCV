# config.py
from pathlib import Path
import torch

# Datasets

DATASET_DFC2020 = "GFM-Bench/DFC2020"  # dùng cho Stage2 segmentation

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

# Stage1: Encoder pretrain on EuroSAT
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
    "input_size": 96,  # DFC2020 patch size
    "batch_size": 64,
    "num_epochs": 1,
    "freeze_encoder_epochs": 5,
    "encoder_lr": 5e-4,
    "decoder_lr": 2e-3,
    "weight_decay": 1e-3,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "min_lr": 1e-6,
    "num_workers": 32,
    "pin_memory": True,
    "mixed_precision": True,
    "gradient_accumulation_steps": 1,
    "save_best_only": True,
    "checkpoint_metric": "loss",
    "log_interval": 50,
    "encoder_checkpoint": STAGE1["save_path"]  # sử dụng pretrain từ Stage1
}

# Problem-specific: DFC2020
NUM_CLASSES = 8  # 8 lớp DFC2020
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

# Normalization placeholders (tính từ dữ liệu thật nếu cần)
SENTINEL2_MEAN = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392, 1263.73947144, 1645.40315151, 1846.87040806,
           1762.59530783, 1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]

SENTINEL2_STD = [633.15169573, 650.2842772, 712.12507725, 965.23119807, 948.9819932, 1108.06650639, 1258.36394548,
          1233.1492281, 1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

SENTINEL1_MEAN = [-12.54847273, -20.19237134]

SENTINEL1_STD = [5.25697717, 5.91150917]
