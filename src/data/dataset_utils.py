# src/data/dataset_utils.py
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import interpolate
import numpy as np
from datasets import load_dataset, DatasetDict

from config import (
    DATASET_DFC2020,
    STAGE2,
    SENTINEL1_MEAN,
    SENTINEL1_STD,
    SENTINEL2_MEAN,
    SENTINEL2_STD
)


# ---------------------------------------
# Stage2: DFC2020 Segmentation (HF Arrow dataset)
# ---------------------------------------
class DFC2020Dataset(Dataset):
    """
    Dataset cho GFM-Bench / DFC2020 segmentation.
    Chỉ lấy 12 kênh chuẩn ResNet50 pretrained: 2 radar + 10 optical
    """

    # index các kênh optical dùng (0-indexed)
    OPTICAL_CHANNELS_10 = [1,2,3,4,5,6,7,8,10,11]  # B02..B08, B8A, B11, B12

    def __init__(self, hf_split, input_size=96):
        self.split = hf_split
        self.input_size = input_size

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        example = self.split[idx]

        # Radar (2 kênh) & Optical (13 kênh)
        radar = np.array(example["radar"], dtype=np.float32)  # (2, H, W)
        optical = np.array(example["optical"], dtype=np.float32)  # (13, H, W)
        label = np.array(example["label"], dtype=np.int64)  # (H, W)

        # Chỉ chọn 10 kênh optical theo pretrained
        optical = optical[self.OPTICAL_CHANNELS_10, :, :]  # (10, H, W)

        # Normalize
        radar = (radar - np.array(SENTINEL1_MEAN)[:, None, None]) / np.array(SENTINEL1_STD)[:, None, None]
        optical = (optical - np.array(SENTINEL2_MEAN)[self.OPTICAL_CHANNELS_10, None, None]) / \
                  np.array(SENTINEL2_STD)[self.OPTICAL_CHANNELS_10, None, None]

        # Convert sang Tensor
        radar_t = torch.from_numpy(radar)
        optical_t = torch.from_numpy(optical)
        label_t = torch.from_numpy(label)

        # Resize về input_size
        if self.input_size is not None:
            radar_t = interpolate(radar_t.unsqueeze(0), size=(self.input_size, self.input_size), mode="bilinear",
                                  align_corners=False).squeeze(0)
            optical_t = interpolate(optical_t.unsqueeze(0), size=(self.input_size, self.input_size), mode="bilinear",
                                    align_corners=False).squeeze(0)
            label_t = interpolate(label_t.unsqueeze(0).unsqueeze(0).float(), size=(self.input_size, self.input_size),
                                  mode="nearest").squeeze(0).long()

        # Concat radar + optical → 12 kênh
        x = torch.cat([radar_t, optical_t], dim=0)  # (12, H', W')

        return {
            "image": x,       # dùng key "image" cho Trainer / utils.py
            "mask": label_t   # (H', W')
        }


def create_dfc2020_loaders(batch_size=None, input_size=None, num_workers=None):
    """
    Load DFC2020 trực tiếp từ HF Arrow dataset mà không cần script cũ.
    """
    batch_size = batch_size or STAGE2["batch_size"]
    input_size = input_size or STAGE2["input_size"]
    num_workers = num_workers or STAGE2["num_workers"]

    # Load Arrow dataset từ HF repo
    ds: DatasetDict = load_dataset(DATASET_DFC2020, trust_remote_code=True)  # Arrow dataset

    train_ds = DFC2020Dataset(ds["train"], input_size=input_size)
    val_ds = DFC2020Dataset(ds["val"], input_size=input_size)
    test_ds = DFC2020Dataset(ds["test"], input_size=input_size)

    print(f"DFC2020 splits sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
