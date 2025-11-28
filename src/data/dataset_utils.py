# src/data/dataset_utils.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import interpolate
import numpy as np
from datasets import load_dataset, DatasetDict

from config import (
    DATASET_DFC2020,
    STAGE2,
    SENTINEL1_MEAN,
    SENTINEL1_STD,
    SENTINEL2_MEAN,
    SENTINEL2_STD,
)

# -------------------------------
# DFC2020 official class mapping
# 0..17 → {0..7,255}
# -------------------------------
DFC2020_CLASSES = np.array([
    255, 0, 0, 0, 0, 0, 1, 1, 255, 255,
    2, 3, 4, 5, 4, 255, 6, 7
], dtype=np.int64)


def map_labels_safe(label_np: np.ndarray):
    mapped = np.full_like(label_np, 255, dtype=np.int64)
    valid_mask = (label_np >= 0) & (label_np < len(DFC2020_CLASSES))
    mapped[valid_mask] = DFC2020_CLASSES[label_np[valid_mask]]

    invalid_mask = (label_np != 255) & ~valid_mask
    if invalid_mask.any():
        uniq = np.unique(label_np[invalid_mask])
        print(f"[WARN] Found INVALID labels (not 0..17 or 255): {uniq}")

    return mapped


class DFC2020Dataset(Dataset):
    OPTICAL_CHANNELS_10 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

    def __init__(self, hf_split, input_size=96, filter_empty=True):
        self.input_size = input_size
        self.raw_split = hf_split
        self.filter_empty = filter_empty

        # ---- Filter examples with all-ignore masks ----
        self.valid_indices = []
        for idx, example in enumerate(self.raw_split):
            label = map_labels_safe(np.array(example["label"], dtype=np.int64))
            # Ignore pixels >7
            label[label > 7] = 255
            if filter_empty:
                if (label != 255).any():
                    self.valid_indices.append(idx)
            else:
                self.valid_indices.append(idx)

        print(f"[INFO] Dataset filtered: {len(self.valid_indices)}/{len(self.raw_split)} examples kept.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        example = self.raw_split[real_idx]

        radar = torch.from_numpy(np.array(example["radar"], dtype=np.float32))
        optical = torch.from_numpy(np.array(example["optical"], dtype=np.float32))[self.OPTICAL_CHANNELS_10]

        radar = (radar - torch.tensor(SENTINEL1_MEAN)[:, None, None]) / torch.tensor(SENTINEL1_STD)[:, None, None]
        optical = (optical - torch.tensor(SENTINEL2_MEAN)[self.OPTICAL_CHANNELS_10, None, None]) / \
                  torch.tensor(SENTINEL2_STD)[self.OPTICAL_CHANNELS_10, None, None]

        if self.input_size is not None:
            radar = interpolate(radar.unsqueeze(0), size=(self.input_size, self.input_size),
                                mode="bilinear", align_corners=False).squeeze(0)
            optical = interpolate(optical.unsqueeze(0), size=(self.input_size, self.input_size),
                                  mode="bilinear", align_corners=False).squeeze(0)

        x = torch.cat([radar, optical], dim=0)

        # ---- Map label ----
        label = torch.from_numpy(map_labels_safe(np.array(example["label"], dtype=np.int64)))
        label[label > 7] = 255
        if self.input_size is not None:
            label = interpolate(label.unsqueeze(0).unsqueeze(0).float(),
                                size=(self.input_size, self.input_size),
                                mode="nearest").squeeze(0).long()
        return {"image": x, "mask": label}


# -------------------------------
# Collate function to create batch
# -------------------------------
def dfc2020_collate_fn(batch):
    imgs = torch.stack([b["image"] for b in batch], dim=0)  # (B,C,H,W)
    masks = torch.stack([b["mask"] for b in batch], dim=0)  # (B,H,W)
    return {"image": imgs, "mask": masks}


def create_dfc2020_loaders(batch_size=None, input_size=None, num_workers=None):
    batch_size = batch_size or STAGE2["batch_size"]
    input_size = input_size or STAGE2["input_size"]
    num_workers = num_workers or STAGE2["num_workers"]

    ds: DatasetDict = load_dataset(DATASET_DFC2020, trust_remote_code=True)

    train_ds = DFC2020Dataset(ds["train"], input_size=input_size)
    val_ds = DFC2020Dataset(ds["val"], input_size=input_size)
    test_ds = DFC2020Dataset(ds["test"], input_size=input_size)

    print(f"DFC2020 splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=dfc2020_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=dfc2020_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             collate_fn=dfc2020_collate_fn)

    return train_loader, val_loader, test_loader
