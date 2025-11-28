# UNet++ Training Guide

## Giới thiệu

UNet++ là kiến trúc segmentation với nested decoder và dense skip connections, cải thiện so với U-Net truyền thống.

**Ưu điểm:**
- Dense skip connections giúp gradient flow tốt hơn
- Nested decoder refine features chi tiết hơn
- Deep supervision giúp training ổn định và accuracy cao hơn
- Hỗ trợ transfer learning từ pretrained encoder

## Cài đặt

```bash
pip install torch torchvision
pip install transformers datasets
pip install matplotlib numpy tqdm
```

## Cách sử dụng

### 1. Training cơ bản

```bash
python training_unetpp.py --backbone resnet50
```

### 2. Training với Deep Supervision (Khuyến nghị)

```bash
python training_unetpp.py --backbone resnet50 --deep_supervision
```

### 3. Training với Pretrained Encoder

```bash
python training_unetpp.py \
    --backbone resnet50 \
    --encoder_checkpoint checkpoints_stage1/best_encoder.pth \
    --deep_supervision
```

### 4. Training với Visualization

```bash
python training_unetpp.py \
    --backbone resnet50 \
    --deep_supervision \
    --visualize
```

## Các tham số

### Backbone Options

- `resnet18` - Nhẹ nhất, nhanh nhất
- `resnet50` - Cân bằng tốt (mặc định)
- `resnet101` - Mạnh nhất, chậm nhất
- `mobilevit` - Tối ưu cho mobile/edge
- `mobilenetv4_hybrid` - Hiệu quả cao

### Các tham số khác

```bash
--backbone BACKBONE              # Chọn backbone (mặc định: resnet50)
--deep_supervision               # Bật deep supervision
--encoder_checkpoint PATH        # Đường dẫn pretrained encoder
--num_classes_encoder N          # Số classes của encoder (mặc định: 19)
--checkpoint_dir DIR             # Thư mục lưu checkpoints
--visualize                      # Tạo visualization sau training
```

## Ví dụ đầy đủ

### ResNet50 với Deep Supervision

```bash
python training_unetpp.py \
    --backbone resnet50 \
    --deep_supervision \
    --encoder_checkpoint checkpoints_stage1/resnet50_encoder.pth \
    --visualize
```

### MobileViT (Nhanh, nhẹ)

```bash
python training_unetpp.py \
    --backbone mobilevit \
    --deep_supervision \
    --visualize
```

### ResNet101 (Accuracy cao nhất)

```bash
python training_unetpp.py \
    --backbone resnet101 \
    --deep_supervision \
    --encoder_checkpoint checkpoints_stage1/resnet101_encoder.pth
```

## Cấu hình Training

Chỉnh sửa `config.py` để thay đổi hyperparameters:

```python
STAGE2 = {
    "batch_size": 16,           # Batch size
    "num_epochs": 50,           # Số epochs
    "encoder_lr": 1e-5,         # Learning rate cho encoder
    "decoder_lr": 1e-4,         # Learning rate cho decoder
    "weight_decay": 1e-4,       # Weight decay
    "input_size": 224,          # Kích thước input
    "num_workers": 4,           # Số workers cho DataLoader
    "mixed_precision": True,    # Mixed precision training
    "log_interval": 50,         # Log mỗi N iterations
    "save_best_only": True,     # Chỉ lưu best model
    "checkpoint_metric": "loss" # Metric để chọn best model
}
```

## Output

Sau khi training, bạn sẽ có:

```
checkpoints_unetpp_{backbone}_ds/  (nếu dùng deep supervision)
├── best_model.pth                 # Best model checkpoint
├── train_history.csv              # Training history
└── visualizations/                # Predictions (nếu dùng --visualize)
    ├── sample_0.png
    ├── sample_1.png
    └── ...
```

Mỗi visualization gồm 5 ảnh:
1. RGB Composite (từ Sentinel-2)
2. NDVI (Normalized Difference Vegetation Index)
3. Radar Composite (từ Sentinel-1)
4. Ground Truth Mask
5. Prediction

## Deep Supervision

Deep supervision train model với multiple outputs ở các độ sâu khác nhau.

**Loss weights:**
- Output 4 (deepest): 1.0
- Output 3: 0.8
- Output 2: 0.6
- Output 1 (shallowest): 0.4

**Khi nào dùng:**
- ✅ Muốn accuracy cao hơn
- ✅ Dataset nhỏ/trung bình
- ✅ Muốn training ổn định hơn
- ✅ Có đủ GPU memory

**Khi nào không dùng:**
- ❌ GPU memory hạn chế
- ❌ Cần inference nhanh nhất
- ❌ Dataset rất lớn

## So sánh Backbones

| Backbone | Parameters | Speed | Accuracy | Use Case |
|----------|-----------|-------|----------|----------|
| ResNet18 | ~11M | ⚡⚡⚡ | ⭐⭐ | Quick experiments |
| ResNet50 | ~25M | ⚡⚡ | ⭐⭐⭐ | **Recommended** |
| ResNet101 | ~44M | ⚡ | ⭐⭐⭐⭐ | Best accuracy |
| MobileViT | ~5M | ⚡⚡⚡ | ⭐⭐⭐ | Mobile/Edge |
| MobileNetV4 | ~6M | ⚡⚡⚡ | ⭐⭐⭐ | Efficient |

## Troubleshooting

### Out of Memory

**Giải pháp:**
```python
# Trong config.py
STAGE2 = {
    "batch_size": 8,      # Giảm từ 16
    "input_size": 192,    # Giảm từ 224
}
```

Hoặc:
```bash
# Dùng backbone nhẹ hơn
python training_unetpp.py --backbone mobilevit

# Tắt deep supervision
python training_unetpp.py --backbone resnet50
```

### Model không hội tụ

**Kiểm tra:**
- Encoder có load đúng không?
- Learning rate có quá cao không?
- Data có normalize đúng không?

**Thử:**
```python
# Giảm learning rate trong config.py
STAGE2 = {
    "encoder_lr": 5e-6,   # Giảm từ 1e-5
    "decoder_lr": 5e-5,   # Giảm từ 1e-4
}
```

### Accuracy thấp

**Cải thiện:**
1. Bật deep supervision
2. Dùng backbone mạnh hơn (ResNet101)
3. Tăng số epochs
4. Dùng pretrained encoder từ Stage-1
5. Kiểm tra class imbalance

### Training chậm

**Tăng tốc:**
1. Tăng batch size (nếu có đủ memory)
2. Dùng backbone nhẹ hơn
3. Giảm input size
4. Tắt deep supervision
5. Giảm num_workers nếu CPU bottleneck

## Workflow khuyến nghị

### Bước 1: Baseline
```bash
python training_unetpp.py --backbone resnet50
```

### Bước 2: Improve với Deep Supervision
```bash
python training_unetpp.py --backbone resnet50 --deep_supervision
```

### Bước 3: Thử backbone khác
```bash
# Nhanh hơn
python training_unetpp.py --backbone mobilevit --deep_supervision

# Chính xác hơn
python training_unetpp.py --backbone resnet101 --deep_supervision
```

### Bước 4: Visualize và Analyze
```bash
python training_unetpp.py \
    --backbone resnet50 \
    --deep_supervision \
    --visualize
```

## Tips

✅ **Luôn dùng pretrained encoder** từ Stage-1 nếu có  
✅ **Bật deep supervision** cho accuracy tốt hơn  
✅ **Monitor cả loss và mIoU** trong training  
✅ **Visualize predictions** để debug  
✅ **Thử nhiều backbone** để tìm best tradeoff  
✅ **Save training history** để so sánh experiments  

## Đọc thêm

- Paper: [UNet++: A Nested U-Net Architecture](https://arxiv.org/abs/1807.10165)
- Project overview: `project-summary-doc.md`
- Quick start: `QUICKSTART.md`

---

Chúc bạn training thành công! 🚀
