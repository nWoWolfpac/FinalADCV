# BigEarthNet → DFC2020 Segmentation Pipeline

Pipeline huấn luyện mô hình DeepLabV3+ (hoặc SegNet) cho bài toán semantic segmentation trên dataset DFC2020, sử dụng encoder pretrained trên BigEarthNet.

## 📋 Mục Lục

- [Tổng Quan](#tổng-quan)
- [Cài Đặt](#cài-đặt)
- [Cấu Trúc Project](#cấu-trúc-project)
- [Hướng Dẫn Training](#hướng-dẫn-training)
- [Hướng Dẫn Evaluation](#hướng-dẫn-evaluation)
- [Cấu Hình](#cấu-hình)
- [Kết Quả](#kết-quả)

---

## 🎯 Tổng Quan

### Mô Tả

Project này implement một pipeline 2-stage để huấn luyện mô hình DeepLabV3+ (hoặc SegNet) cho semantic segmentation trên dataset DFC2020:

1. **Stage 1**: Pretrain encoder trên BigEarthNet (19 classes classification)
2. **Stage 2**: Fine-tune DeepLabV3+ (hoặc SegNet) trên DFC2020 (8 classes segmentation)

### Dataset

- **DFC2020**: Semantic segmentation dataset với 8 classes
  - Input: 12 channels (2 radar + 10 optical từ Sentinel-1 và Sentinel-2)
  - Output: Segmentation mask với 8 classes
  - Repository: `GFM-Bench/DFC2020` trên HuggingFace

### Model Architecture

- **Encoder**: Resnet18/ResNet50/ResNet101 pretrained trên BigEarthNet
- **Decoder**: DeepLabV3+ (hoặc SegNet) với skip connections
- **Input**: 12-channel images (96×96)
- **Output**: 8-class segmentation masks

---

## 🚀 Cài Đặt

### Yêu Cầu Hệ Thống

- Python >= 3.8
- CUDA-capable GPU (khuyến nghị 8GB+ VRAM)
- 16GB+ RAM

### Cài Đặt Dependencies

```bash
# Clone repository
!git clone https://github.com/nWoWolfpac/FinalADCV.git
cd FinalADCV

# Cài đặt packages
pip install -r requirements.txt
```

### Cài Đặt HuggingFace Dataset

Dataset sẽ tự động được tải về từ HuggingFace khi chạy training lần đầu. Đảm bảo bạn đã đăng nhập HuggingFace:

```bash
huggingface-cli login
```

---

## 📁 Cấu Trúc Project

```
FinalADCV/
├── config.py                 # Cấu hình chính (hyperparameters, paths)
├── training_deeplabv3plus.py          # Script training DeepLabV3+
├── training_segnet.py # Script training SegNet
├── evaluation.py          # Script evaluation
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── models/
│   │   ├── encoder.py                   # Encoder pretrained trên BigEarthNet
│   │   └── deeplabv3plus.py          # DeepLabV3+ architecture
│   │   └── segnet.py          # Segnet architecture       
│   ├── data/
│   │   └── dataset_utils.py # Dataset loading và preprocessing
│   └── utils.py             # Trainer, metrics, visualization
│
├── checkpoints/             # Model checkpoints (tự động tạo)
├── logs/                    # Training logs (tự động tạo)
├── visualizations/          # Prediction visualizations (tự động tạo)
│
└── README.md               # File này
```

---

## 🏋️ Hướng Dẫn Training

### Training Cơ Bản

```bash
python training_deeplabv3plus.py --backbone {Backbone} (Nếu muốn sử dụng mô hình DeepLabV3+)
hoặc
python training_segnet.py --backbone {Backbone} (Nếu muốn sử dụng mô hình SegNet)
```

### Các Tham Số Training

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--backbone` | Encoder backbone (resnet,18 resnet50, resnet101) | resnet50 |
| `--dropout` | Dropout rate cho decoder | 0.1 |
| `--resume` | Đường dẫn checkpoint để resume training | None |

### Thay Đổi Input Size

Để training với input size khác (128×128 hoặc 256×256), sửa trong `config.py`:

```python
STAGE2 = {
    "input_size": 128,        # Thay đổi từ 96
    "batch_size": 12,          # Giảm batch_size tương ứng
    # ... các config khác
}
```

**Lưu ý về Batch Size:**
- `input_size=96`: `batch_size=16` (GPU 8GB+)
- `input_size=128`: `batch_size=12` (GPU 8GB+)
- `input_size=256`: `batch_size=8` (GPU 16GB+)

### Training Process

1. **Load Dataset**: Tự động tải từ HuggingFace
2. **Load Encoder**: Load pretrained encoder từ BigEarthNet
3. **Freeze Encoder**: Freeze encoder trong 5 epochs đầu
4. **Fine-tune**: Unfreeze và fine-tune toàn bộ model
5. **Save Checkpoints**: Lưu best model dựa trên validation loss

### Output Files

Sau khi training, các file sau sẽ được tạo:

```
checkpoints/
├── best_model.pth           # Best model checkpoint
├── checkpoint_epoch_XX.pth  # Checkpoints theo epoch
└── train_history.csv        # Training history (loss, metrics)
```

---

## 📊 Hướng Dẫn Evaluation

### Evaluation trên Test Set

```bash
# Evaluate với best model
python evaluation.py \
    --model {model_name} \
    --backbone {backbone} \
    --checkpoint {checkpoint_dir} \
    --num_classes 8
```
Với model_name = {deeplabv3,segnet} là mô hình đã sử dụng khi training 
    backbone = {resnet18, resnet50, resnet101} là backbone đã sử dụng khi training 
    checkpoint_dir là đường dẫn tới checkpoint sau khi huấn luyện

### Metrics Được Tính

- **Pixel Accuracy**: Tỷ lệ pixels được phân loại đúng
- **Mean IoU (mIoU)**: Intersection over Union trung bình
- **Per-class IoU**: IoU cho từng class

### Visualization

Script sẽ tự động tạo visualizations trong thư mục `visualizations/`:

- RGB composite từ Sentinel-2
- NDVI (Normalized Difference Vegetation Index)
- Radar composite (VV, VH)
- Ground Truth mask
- Predicted mask

---

## ⚙️ Cấu Hình

### File `config.py`

File cấu hình chính chứa tất cả hyperparameters:

```python
# Stage2: Segmentation on DFC2020
STAGE2 = {
    "input_size": 96,               # Input resolution
    "batch_size": 256,               # Batch size
    "num_epochs": 50,                # Số epochs
    "freeze_encoder_epochs": 5,      # Số epochs freeze encoder
    "encoder_lr": 1e-4,              # Learning rate cho encoder
    "decoder_lr": 5e-5,              # Learning rate cho decoder
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "mixed_precision": True,         # Sử dụng mixed precision training
    "gradient_accumulation_steps": 1,
    # ...
}
```

### Normalization Values

Dataset được normalize với các giá trị đã tính từ training set:

```python
SENTINEL1_MEAN = [-12.190531, -19.398623]
SENTINEL1_STD = [5.172539, 6.659642]

SENTINEL2_MEAN = [995.894858, 901.027080, ...]  # 10 channels
SENTINEL2_STD = [257.763536, 299.047602, ...]   # 10 channels
```

---

## 📈 Kết Quả

### Training Metrics

Training history được lưu trong `train_history.csv` với các cột:

- `epoch`: Số epoch
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `pixel_accuracy`: Pixel accuracy trên validation set
- `mean_iou`: Mean IoU trên validation set


## 🎓 Ví Dụ Sử Dụng

### 1. Training từ đầu

```bash
# Step 1: Training với ResNet18
python training_deeplabv3plus.py --backbone resnet18 

Với Backbone = {resnet18, resnet50, resnet101}

# Step 2: Sau khi training xong, evaluate
python evaluation.py \
    --model deeplabv3 \
    --backbone resnet50 \
    --checkpoint {checkpoint_resnet50_dir} \
    --num_classes 8
```



