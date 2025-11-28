# Quick Start Guide - UNet++

## Bắt đầu nhanh với UNet++

### 1. Cài đặt dependencies

```bash
pip install torch torchvision
pip install transformers datasets
pip install matplotlib numpy tqdm pandas
```

### 2. Chuẩn bị dữ liệu

Đảm bảo dataset DFC2020 đã được tải và đặt đúng đường dẫn trong `config.py`:

```python
# config.py
STAGE2 = {
    "dataset_path": "path/to/dfc2020",  # Cập nhật đường dẫn này
    ...
}
```

### 3. Train UNet++

#### Option A: ResNet50 + Deep Supervision (Khuyến nghị)

```bash
python training_unetpp.py \
    --backbone resnet50 \
    --deep_supervision \
    --visualize
```

#### Option B: MobileViT (Nhanh, nhẹ)

```bash
python training_unetpp.py \
    --backbone mobilevit \
    --deep_supervision \
    --visualize
```

#### Option C: ResNet101 (Accuracy cao nhất)

```bash
python training_unetpp.py \
    --backbone resnet101 \
    --deep_supervision \
    --visualize
```

### 4. Kiểm tra kết quả

Sau khi training xong, kiểm tra:

```
checkpoints_unetpp_{backbone}_ds/
├── best_model.pth              # Model tốt nhất
├── train_history.csv           # Lịch sử training
└── visualizations/             # Ảnh kết quả (nếu dùng --visualize)
```

### 5. Chạy nhiều experiments

```bash
# Linux/Mac
bash run_experiments.sh

# Windows
run_experiments.bat
```

### 6. So sánh kết quả

```bash
python compare_results.py
```

## Các lệnh thường dùng

### Train với pretrained encoder

```bash
python training_unetpp.py \
    --backbone resnet50 \
    --encoder_checkpoint checkpoints_stage1/best_encoder.pth \
    --deep_supervision \
    --visualize
```

### Train với backbone khác

```bash
# ResNet18 (nhẹ nhất)
python training_unetpp.py --backbone resnet18 --deep_supervision

# ResNet101 (mạnh nhất)
python training_unetpp.py --backbone resnet101 --deep_supervision

# MobileViT (cân bằng)
python training_unetpp.py --backbone mobilevit --deep_supervision

# MobileNetV4 (hiệu quả)
python training_unetpp.py --backbone mobilenetv4_hybrid --deep_supervision
```

### Điều chỉnh hyperparameters

Sửa file `config.py`:

```python
STAGE2 = {
    "batch_size": 16,           # Giảm nếu out of memory
    "num_epochs": 50,           # Tăng để train lâu hơn
    "encoder_lr": 1e-5,         # Learning rate cho encoder
    "decoder_lr": 1e-4,         # Learning rate cho decoder
    "input_size": 224,          # Kích thước input
    ...
}
```

## Troubleshooting nhanh

### Out of Memory?
```python
# Trong config.py
STAGE2 = {
    "batch_size": 8,  # Giảm từ 16
    "input_size": 192,  # Giảm từ 224
}
```

### Model không hội tụ?
- Kiểm tra learning rate (giảm xuống)
- Bật deep supervision cho UNet++
- Tăng số epochs

### Muốn train nhanh hơn?
- Dùng backbone nhẹ hơn (resnet18, mobilevit)
- Tắt deep supervision
- Giảm input size
- Tăng batch size (nếu có đủ memory)

### Muốn accuracy cao hơn?
- Bật deep supervision
- Dùng backbone mạnh hơn (resnet101)
- Tăng số epochs
- Dùng pretrained encoder từ Stage-1

## Workflow khuyến nghị

1. **Baseline**: Bắt đầu với ResNet50 không deep supervision
   ```bash
   python training_unetpp.py --backbone resnet50
   ```

2. **Improve**: Bật deep supervision
   ```bash
   python training_unetpp.py --backbone resnet50 --deep_supervision --visualize
   ```

3. **Optimize**: Thử backbone khác nhau
   ```bash
   # Nhanh hơn
   python training_unetpp.py --backbone mobilevit --deep_supervision
   
   # Chính xác hơn
   python training_unetpp.py --backbone resnet101 --deep_supervision
   ```

4. **Compare**: So sánh kết quả
   ```bash
   python compare_results.py
   ```

## Chạy tất cả experiments

```bash
# Linux/Mac
bash run_experiments.sh

# Windows
run_experiments.bat

# Sau đó so sánh
python compare_results.py
```

## Đọc thêm

- Chi tiết về UNet++: `README_UNETPP.md`
- Tổng quan project: `project-summary-doc.md`
- So sánh experiments: `python compare_results.py`

## Tips cuối cùng

✅ Luôn dùng pretrained encoder từ Stage-1  
✅ Bật deep supervision cho accuracy tốt hơn  
✅ Monitor cả loss và mIoU  
✅ Visualize predictions để debug  
✅ Thử nhiều backbone để tìm best tradeoff  
✅ Dùng `compare_results.py` để so sánh experiments  

Chúc bạn train thành công! 🚀
