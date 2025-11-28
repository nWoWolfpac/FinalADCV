# UNet++ for Satellite Image Segmentation

Semantic segmentation cho ảnh vệ tinh sử dụng UNet++ với transfer learning.

## 📋 Tổng quan

- **Task**: Semantic Segmentation
- **Input**: 12-band satellite imagery (Sentinel-1 + Sentinel-2)
- **Output**: 8-class land cover segmentation
- **Model**: UNet++ với nested decoder và dense skip connections
- **Transfer Learning**: 2-stage approach với pretrained encoder

## 🚀 Quick Start

### Option A: Local Training

#### 1. Cài đặt

```bash
pip install torch torchvision transformers datasets
pip install matplotlib numpy tqdm pandas
```

#### 2. Train

```bash
# Khuyến nghị: ResNet50 + Deep Supervision
python training_unetpp.py --backbone resnet50 --deep_supervision --visualize
```

### Option B: JetBrains Cadence (Cloud GPU) ☁️

**Chạy trên GPU cloud trực tiếp từ PyCharm!**

1. Mở PyCharm → `Tools` → `Cadence`
2. Đăng nhập JetBrains Account
3. Chọn run configuration: "Train UNet++ ResNet50"
4. Click "Run on Cadence" ☁️
5. Chọn GPU (T4/A10/A100) và Start!

📖 **Chi tiết**: Xem `CADENCE_GUIDE.md`

### 3. Xem kết quả

```
checkpoints_unetpp_resnet50_ds/
├── best_model.pth
├── train_history.csv
└── visualizations/
```

## 📁 Cấu trúc Project

```
FinalADCV/
├── training_unetpp.py          # Main training script
├── config.py                   # Configuration
├── compare_results.py          # Compare experiments
├── run_experiments.sh/.bat     # Run multiple experiments
│
├── src/
│   ├── models/
│   │   ├── encoder.py          # Pretrained encoder
│   │   └── unetplusplus.py     # UNet++ implementation
│   ├── data/
│   │   └── dataset_utils.py    # DFC2020 dataset
│   └── utils.py                # Training utilities
│
├── README.md                   # This file
├── README_UNETPP.md           # Detailed UNet++ guide
├── QUICKSTART.md              # Quick start guide
└── project-summary-doc.md     # Project overview
```

## 🎯 Các tính năng

### UNet++ Model
- ✅ Nested decoder với dense skip connections
- ✅ Deep supervision (optional)
- ✅ Multiple backbone support (ResNet, MobileViT, MobileNetV4)
- ✅ Transfer learning từ BigEarthNet pretrained encoder
- ✅ 12-band input support

### Training
- ✅ Mixed precision training
- ✅ Differential learning rates (encoder vs decoder)
- ✅ Automatic checkpointing
- ✅ Training history logging
- ✅ Visualization generation

### Evaluation
- ✅ mIoU (mean Intersection over Union)
- ✅ Pixel accuracy
- ✅ Per-class metrics
- ✅ Confusion matrix

## 🔧 Cách sử dụng

### Training cơ bản

```bash
python training_unetpp.py --backbone resnet50 --deep_supervision
```

### Training với pretrained encoder

```bash
python training_unetpp.py \
    --backbone resnet50 \
    --encoder_checkpoint checkpoints_stage1/best_encoder.pth \
    --deep_supervision \
    --visualize
```

### Chạy nhiều experiments

```bash
# Linux/Mac
bash run_experiments.sh

# Windows
run_experiments.bat
```

### So sánh kết quả

```bash
python compare_results.py
```

## 📊 Backbones

| Backbone | Parameters | Speed | Accuracy | Khuyến nghị |
|----------|-----------|-------|----------|-------------|
| ResNet18 | ~11M | ⚡⚡⚡ | ⭐⭐ | Quick experiments |
| **ResNet50** | ~25M | ⚡⚡ | ⭐⭐⭐ | **Recommended** |
| ResNet101 | ~44M | ⚡ | ⭐⭐⭐⭐ | Best accuracy |
| MobileViT | ~5M | ⚡⚡⚡ | ⭐⭐⭐ | Mobile/Edge |
| MobileNetV4 | ~6M | ⚡⚡⚡ | ⭐⭐⭐ | Efficient |

## 🎓 Deep Supervision

Deep supervision train model với multiple outputs ở các độ sâu khác nhau.

**Khi nào dùng:**
- ✅ Muốn accuracy cao hơn
- ✅ Dataset nhỏ/trung bình
- ✅ Có đủ GPU memory

**Loss weights:**
- Output 4 (deepest): 1.0
- Output 3: 0.8
- Output 2: 0.6
- Output 1 (shallowest): 0.4

## ⚙️ Configuration

Chỉnh sửa `config.py`:

```python
STAGE2 = {
    "batch_size": 16,           # Batch size
    "num_epochs": 50,           # Số epochs
    "encoder_lr": 1e-5,         # LR cho encoder
    "decoder_lr": 1e-4,         # LR cho decoder
    "input_size": 224,          # Input size
    "mixed_precision": True,    # Mixed precision
}
```

## 🐛 Troubleshooting

### Out of Memory
```python
# Giảm batch size và input size trong config.py
STAGE2 = {
    "batch_size": 8,
    "input_size": 192,
}
```

### Model không hội tụ
- Giảm learning rate
- Bật deep supervision
- Kiểm tra data normalization

### Accuracy thấp
- Bật deep supervision
- Dùng backbone mạnh hơn
- Tăng số epochs
- Dùng pretrained encoder

## 📚 Documentation

- **README_UNETPP.md** - Chi tiết về UNet++ và cách sử dụng
- **QUICKSTART.md** - Hướng dẫn bắt đầu nhanh
- **project-summary-doc.md** - Tổng quan về project

## 🔬 Experiments

Chạy tất cả experiments và so sánh:

```bash
# Chạy 4 experiments với các cấu hình khác nhau
bash run_experiments.sh  # hoặc run_experiments.bat trên Windows

# So sánh kết quả
python compare_results.py
```

Kết quả sẽ được lưu trong `comparison_results/`:
- `summary.csv` - Bảng tổng hợp
- `*_comparison.png` - Biểu đồ so sánh
- `combined_comparison.png` - Biểu đồ tổng hợp

## 📖 References

- **UNet++**: [A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)
- **BigEarthNet**: [BigEarthNet Dataset](https://bigearth.net/)
- **DFC2020**: [IEEE GRSS Data Fusion Contest 2020](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest)

## 📝 Citation

```bibtex
@article{zhou2018unetplusplus,
  title={UNet++: A Nested U-Net Architecture for Medical Image Segmentation},
  author={Zhou, Zongwei and Siddiquee, Md Mahfuzur Rahman and Tajbakhsh, Nima and Liang, Jianming},
  journal={arXiv preprint arXiv:1807.10165},
  year={2018}
}
```

## 📄 License

This project is for educational purposes.

---

**Chúc bạn training thành công! 🚀**

Nếu có vấn đề, xem thêm tại `README_UNETPP.md` hoặc `QUICKSTART.md`
