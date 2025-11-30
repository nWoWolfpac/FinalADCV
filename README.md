# UNet++ Land Cover Segmentation

Semantic segmentation cho ảnh vệ tinh Sentinel-2 sử dụng kiến trúc UNet++ với transfer learning.

## Tổng quan

- **Input**: Ảnh 12 band (10 Sentinel-2 optical + 2 Sentinel-1 SAR)
- **Output**: Segmentation mask 8 classes
- **Dataset**: DFC2020 (IEEE GRSS Data Fusion Contest 2020)
- **Model**: UNet++ với pretrained encoder

## 8 Classes

| ID | Class | Mô tả |
|----|-------|-------|
| 0 | Forest | Rừng |
| 1 | Shrubland | Cây bụi |
| 2 | Grassland | Đồng cỏ |
| 3 | Wetlands | Đất ngập nước |
| 4 | Croplands | Đất nông nghiệp |
| 5 | Urban/Built-up | Đô thị |
| 6 | Barren | Đất trống |
| 7 | Water | Nước |

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách chạy

### Training UNet++

```bash
# Cơ bản
python training_unetpp.py --backbone resnet50

# Với deep supervision (khuyến nghị)
python training_unetpp.py --backbone resnet50 --deep_supervision --visualize
```

### Backbones hỗ trợ
 Backbone 
----------
 resnet18 
 resnet50 
 resnet101
 mobilevit 

### Evaluation

```bash
python evaluation.py --backbone resnet50 --deep_supervision \
    --checkpoint checkpoints_unetpp_resnet50_ds/best_model.pth
```

## Cấu trúc Project

```
FinalADCV/
├── config.py                 # Cấu hình training
├── training_unetpp.py        # Script train UNet++
├── training_decoder.py       # Script train DeepLabV3+
├── evaluation.py             # Đánh giá model
├── compare_results.py        # So sánh experiments
├── src/
│   ├── data/
│   │   └── dataset_utils.py  # Data loading
│   ├── models/
│   │   ├── encoder.py        # Pretrained encoders
│   │   ├── unetplusplus.py   # UNet++ model
│   │   └── deeplabv3plus.py  # DeepLabV3+ model
│   └── utils.py              # Trainer, metrics
└── checkpoints_*/            # Saved models
```

## Output

Sau khi train xong:

```
checkpoints_unetpp_resnet50_ds/
├── best_model.pth          # Best checkpoint
├── train_history.csv       # Training metrics
└── visualizations/         # Sample predictions
    ├── sample_0.png
    └── ...
```

## Chạy trên Kaggle

Script tự động detect Kaggle và lưu vào `/kaggle/working/`:

```python
# Kaggle notebook
!python training_unetpp.py --backbone resnet50 --deep_supervision --visualize
```

## Cấu hình

Sửa `config.py` để thay đổi hyperparameters:

```python
STAGE2 = {
    "input_size": 96,
    "batch_size": 32,
    "num_epochs": 50,
    "encoder_lr": 1e-5,
    "decoder_lr": 1e-4,
}
```

## Deep Supervision

UNet++ hỗ trợ deep supervision với 4 outputs:
- Weights: [1.0, 0.8, 0.6, 0.4] (deepest → shallowest)
- Giúp model hội tụ nhanh hơn và tránh overfitting

## References

- [UNet++: A Nested U-Net Architecture](https://arxiv.org/abs/1807.10165)
- [DFC2020 Dataset](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest)
- [BigEarthNet](https://bigearth.net/)
