# Bài toán: Phân loại Loại hình Sử dụng Đất từ Ảnh Vệ tinh sử dụng U-Net

## 1. MÔ TẢ BÀI TOÁN

### 1.1. Tổng quan
**Bài toán**: Image Semantic Segmentation cho ảnh vệ tinh
- **Đầu vào**: Ảnh 12 band từ vệ tinh Sentinel-2
- **Đầu ra**: Segmentation mask phân loại từng pixel theo loại hình sử dụng đất
- **Số classes**: 8 classes
- **Kiến trúc**: Các model transfer learning

### 1.2. Phương pháp tiếp cận
**Chiến lược Transfer Learning 2-giai đoạn**:

**Giai đoạn 1 - Pre-training Encoder**:
- Sử dụng pre-trained model **BigEarthNet**
- Task: Classification với 19 classes
- Mục tiêu: Học các feature representations tốt cho ảnh vệ tinh

**Giai đoạn 2 - Fine-tuning toàn bộ **:
- Sử dụng tập **DFC2020** để train toàn bộ
- Task: Semantic Segmentation
- Encoder được khởi tạo từ weights đã pre-train
- Decoder được train từ đầu
- Lợi ích: Hội tụ nhanh hơn, performance tốt hơn


### 2. Cấu trúc dự án
```

project/\
│\
├── config.py\
├── training_decoder.py\
├── project_summary.md\
│\
├── src/\
│ ├── data/\
│ │ └── dataset_utils.py\
│ │\
│ ├── models/\
│ │ ├── encoder.py\
│ │ ├── deeplabv3plus.py\
│ │ └── unetplusplus.py\
│ │\
│ ├── utils.py\
│\
├── README_MODELS.md
│\
├── checkpoints_stage1/\
├── checkpoints_stage2/\
└── datasets/

```

---

## 3. Mô tả chi tiết từng file

### 3.1 File cấp root

#### **config.py**
- Chứa toàn bộ cấu hình của dự án:
  - Device (GPU/CPU)
  - Learning rate
  - Batch size
  - Class numbers
  - Đường dẫn dataset
  - Backbone mặc định
  - Cấu hình Stage-1 & Stage-2
- Chỉ cần sửa file này để điều chỉnh training.


#### **train_stage2.py**
- Huấn luyện segmentation.
- Nhiệm vụ:
  - Load encoder Stage-1.
  - Khởi tạo DeepLabV3+ với encoder đó.
  - Tải dataset DFC2020 (12 band).
  - Train segmentation.
  - Lưu mô hình tốt nhất vào `checkpoints_stage2/`.
  - Xuất ảnh visualize.

---

## 3.2 Module `src/data/dataset_utils.py`

### **dfc2020_dataset.py**
- Xử lý bộ dữ liệu DFC2020/SEN12MS.
- Load:
  - Optical Sentinel-2: 10 band
  - SAR Sentinel-1: 2 band
- Ghép thành tensor (12, H, W).
- Trả về:
```

{\
"image": tensor(12,H,W),\
"mask": tensor(H,W)\
}

```
- Hỗ trợ transforms/augmentations.
- Tạo DataLoader cho DFC2020:
```

train_loader, val_loader, test_loader = create_dfc2020_loaders(...)

```
- Tự chia theo tỷ lệ chuẩn.
- Hỗ trợ shuffle, num_workers, pin_memory.

---

## 3.3 Module `src/models/`

### **encoder.py**
- Tập trung xử lý backbone HuggingFace:
- resnet18 / resnet50 / resnet101 (BigEarthNet pretrained)
- mobilevit-small
- mobilenetv4_hybrid
- Tính năng:
- **Giữ nguyên số lượng band**
- Tự sửa conv đầu vào khi số band ≠ 3.
- Hỗ trợ freezing hoặc fine-tuning.
- Trả về encoder + số channels output.

### **deeplabv3plus.py**
- Cài đặt đầy đủ mô hình DeepLabV3+:
- ASPP  
- Decoder head  
- Skip connection  
- Nhận encoder từ Stage-1.
- Hỗ trợ input_channels = 12.

### **unetplusplus.py**
- Cài đặt đầy đủ mô hình UNet++:
- Nested decoder với dense skip connections
- 5 encoder stages (enc0-enc4)
- Nested decoder blocks (dec0_1, dec0_2, dec0_3, dec0_4, etc.)
- Hỗ trợ deep supervision (multiple outputs)
- Nhận encoder từ Stage-1
- Hỗ trợ input_channels = 12


---

## 3.4 Module `src/utils.py`

- Lớp huấn luyện tổng quát (Stage-1 và Stage-2).
- Hỗ trợ:
- AMP (mixed precision)
- Gradient accumulation
- Checkpointing
- Early stopping
- Log loss/mIoU
- LR riêng cho encoder/decoder

### **metrics.py**
- Tính:
- mIoU
- pixel accuracy
- precision / recall
- F1-score
- confusion matrix

---

## 4. Pipeline hoạt động

### **Stage-1: Pretraining BigEarthNet**
1. Load backbone pretrained từ HuggingFace.  
2. Điều chỉnh conv đầu vào nếu số band ≠ 3.  
3. Train/Fine-tune classification.  
4. Lưu encoder đã huấn luyện.

### **Stage-2: Segmentation DFC2020**
1. Load encoder Stage-1.  
2. Khởi tạo DeepLabV3+.  
3. Dataset: 12 band (10 optical + 2 SAR).  
4. Train segmentation.  
5. Lưu best checkpoint + visualization.

---

## 5. Cách chạy

### 5.1. DeepLabV3+
```bash
python training_decoder.py --model deeplabv3plus --backbone resnet50
```

### 5.2. UNet++
```bash
# Không deep supervision
python training_decoder.py --model unetplusplus --backbone resnet50

# Với deep supervision (khuyến nghị)
python training_decoder.py --model unetplusplus --backbone resnet50 --deep_supervision
```

### 5.3. Các backbone khác
```bash
# ResNet18
python training_decoder.py --model unetplusplus --backbone resnet18

# ResNet101
python training_decoder.py --model deeplabv3plus --backbone resnet101

# MobileViT
python training_decoder.py --model unetplusplus --backbone mobilevit --deep_supervision

# MobileNetV4
python training_decoder.py --model deeplabv3plus --backbone mobilenetv4_hybrid
```

### 5.4. Với pretrained encoder
```bash
python training_decoder.py \
    --model unetplusplus \
    --backbone resnet50 \
    --encoder_checkpoint checkpoints_stage1/best_encoder.pth \
    --deep_supervision
```
---

## 6. Các model đã implement

### 6.1. DeepLabV3+
- ✅ ASPP module với multi-scale context
- ✅ Decoder với skip connection
- ✅ Hỗ trợ 3 backbone: ResNet, MobileViT, MobileNetV4
- ✅ Transfer learning từ BigEarthNet

### 6.2. UNet++
- ✅ Nested U-Net với dense skip connections
- ✅ Deep supervision (optional)
- ✅ Hỗ trợ 3 backbone: ResNet, MobileViT, MobileNetV4
- ✅ Transfer learning từ BigEarthNet

## 7. Mở rộng trong tương lai

- Thử nghiệm mô hình khác: UNet, FPN, SegFormer, Mask2Former.
- Pseudo-labeling tăng dữ liệu.
- Tiled inference cho ảnh rất lớn.
- Export ONNX hoặc TensorRT để tăng tốc.
- Test-time augmentation (TTA).
- Ensemble multiple models.

---

```