# UNet++ Project Checklist

## ✅ Files đã tạo

### Core Implementation
- [x] `src/models/unetplusplus.py` - UNet++ model implementation
- [x] `src/models/encoder.py` - Pretrained encoder (đã có sẵn)
- [x] `src/data/dataset_utils.py` - Dataset utilities (đã có sẵn)
- [x] `src/utils.py` - Training utilities (đã cập nhật)

### Training Scripts
- [x] `training_unetpp.py` - Main training script cho UNet++
- [x] `config.py` - Configuration file (đã có sẵn)

### Experiment Tools
- [x] `run_experiments.sh` - Bash script để chạy nhiều experiments (Linux/Mac)
- [x] `run_experiments.bat` - Batch script để chạy nhiều experiments (Windows)
- [x] `compare_results.py` - Script so sánh kết quả experiments

### Documentation
- [x] `README.md` - Main README
- [x] `README_UNETPP.md` - Chi tiết về UNet++
- [x] `QUICKSTART.md` - Quick start guide
- [x] `project-summary-doc.md` - Project overview (đã cập nhật)
- [x] `CHECKLIST.md` - File này

## 📋 Trước khi train

### 1. Kiểm tra môi trường
```bash
# Kiểm tra Python
python --version  # >= 3.8

# Kiểm tra PyTorch
python -c "import torch; print(torch.__version__)"

# Kiểm tra CUDA (nếu dùng GPU)
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Cài đặt dependencies
```bash
pip install torch torchvision
pip install transformers datasets
pip install matplotlib numpy tqdm pandas
```

### 3. Chuẩn bị dataset
- [ ] Download DFC2020 dataset
- [ ] Giải nén dataset
- [ ] Cập nhật đường dẫn trong `config.py`

```python
# config.py
STAGE2 = {
    "dataset_path": "path/to/dfc2020",  # ← Cập nhật đường dẫn này
    ...
}
```

### 4. (Optional) Chuẩn bị pretrained encoder
- [ ] Train encoder ở Stage-1 (hoặc dùng pretrained từ HuggingFace)
- [ ] Lưu encoder weights
- [ ] Ghi nhớ đường dẫn để dùng với `--encoder_checkpoint`

## 🚀 Training Workflow

### Chọn môi trường training:

**Option A: Local**
```bash
python training_unetpp.py --backbone resnet50
```

**Option B: Cadence (Cloud GPU)** ☁️
1. Mở PyCharm
2. Chọn run configuration "Train UNet++ ResNet50"
3. Click "Run on Cadence"
4. Chọn GPU và Start

📖 Chi tiết: `CADENCE_GUIDE.md`

### Bước 1: Baseline
```bash
python training_unetpp.py --backbone resnet50
```
- [ ] Chạy thành công
- [ ] Kiểm tra output trong `checkpoints_unetpp_resnet50/`
- [ ] Xem `train_history.csv`

### Bước 2: Deep Supervision
```bash
python training_unetpp.py --backbone resnet50 --deep_supervision --visualize
```
- [ ] Chạy thành công
- [ ] So sánh với baseline
- [ ] Kiểm tra visualizations

### Bước 3: Thử backbone khác
```bash
# MobileViT (nhanh)
python training_unetpp.py --backbone mobilevit --deep_supervision

# ResNet101 (chính xác)
python training_unetpp.py --backbone resnet101 --deep_supervision
```
- [ ] Chạy với MobileViT
- [ ] Chạy với ResNet101
- [ ] So sánh kết quả

### Bước 4: Experiments
```bash
# Chạy tất cả experiments
bash run_experiments.sh  # hoặc run_experiments.bat

# So sánh kết quả
python compare_results.py
```
- [ ] Chạy experiments
- [ ] Xem comparison plots
- [ ] Chọn best model

## 📊 Sau khi train

### 1. Kiểm tra kết quả
- [ ] Xem `train_history.csv`
- [ ] Kiểm tra best mIoU
- [ ] Kiểm tra pixel accuracy
- [ ] Xem visualizations

### 2. So sánh experiments
- [ ] Chạy `compare_results.py`
- [ ] Xem `comparison_results/summary.csv`
- [ ] Xem comparison plots
- [ ] Chọn best configuration

### 3. Đánh giá model
- [ ] Load best checkpoint
- [ ] Test trên test set
- [ ] Tính metrics chi tiết
- [ ] Visualize predictions

## 🐛 Troubleshooting Checklist

### Out of Memory
- [ ] Giảm batch_size trong config.py
- [ ] Giảm input_size trong config.py
- [ ] Dùng backbone nhẹ hơn (mobilevit)
- [ ] Tắt deep supervision
- [ ] Giảm num_workers

### Model không hội tụ
- [ ] Kiểm tra learning rate
- [ ] Kiểm tra data normalization
- [ ] Bật deep supervision
- [ ] Kiểm tra encoder có load đúng không
- [ ] Tăng số epochs

### Accuracy thấp
- [ ] Bật deep supervision
- [ ] Dùng backbone mạnh hơn
- [ ] Dùng pretrained encoder
- [ ] Tăng số epochs
- [ ] Kiểm tra class imbalance
- [ ] Thử data augmentation

### Training chậm
- [ ] Tăng batch size
- [ ] Dùng backbone nhẹ hơn
- [ ] Giảm input size
- [ ] Tắt deep supervision
- [ ] Bật mixed precision
- [ ] Giảm num_workers nếu CPU bottleneck

## 📝 Notes

### Best Practices
- ✅ Luôn dùng pretrained encoder từ Stage-1
- ✅ Bật deep supervision cho accuracy tốt hơn
- ✅ Monitor cả loss và mIoU
- ✅ Visualize predictions để debug
- ✅ Save training history
- ✅ Compare multiple experiments

### Recommended Settings
```python
# config.py - Recommended for most cases
STAGE2 = {
    "batch_size": 16,           # Giảm nếu OOM
    "num_epochs": 50,           # Tăng nếu cần
    "encoder_lr": 1e-5,         # Thấp hơn decoder
    "decoder_lr": 1e-4,         # Cao hơn encoder
    "weight_decay": 1e-4,
    "input_size": 224,
    "mixed_precision": True,    # Tăng tốc training
}
```

### Experiment Tracking
Tạo bảng để track experiments:

| Exp | Backbone | Deep Sup | Best mIoU | Best Acc | Notes |
|-----|----------|----------|-----------|----------|-------|
| 1   | resnet50 | No       |           |          |       |
| 2   | resnet50 | Yes      |           |          |       |
| 3   | mobilevit| Yes      |           |          |       |
| 4   | resnet101| Yes      |           |          |       |

## 🎯 Goals

### Minimum Goals
- [ ] Train UNet++ thành công
- [ ] Đạt mIoU > 0.5
- [ ] Generate visualizations

### Target Goals
- [ ] So sánh ít nhất 3 backbones
- [ ] Đạt mIoU > 0.6
- [ ] Deep supervision improve accuracy

### Stretch Goals
- [ ] Đạt mIoU > 0.7
- [ ] Optimize inference speed
- [ ] Export model to ONNX

## ✨ Hoàn thành!

Khi đã hoàn thành tất cả:
- [ ] Có best model với mIoU cao
- [ ] Có comparison results
- [ ] Có visualizations
- [ ] Hiểu rõ tradeoffs giữa các backbones
- [ ] Document kết quả

---

**Good luck! 🚀**
