# Hướng dẫn chạy UNet++ trên JetBrains Cadence

## 📋 Tổng quan

JetBrains Cadence cho phép bạn chạy training trên GPU cloud trực tiếp từ PyCharm mà không cần setup server riêng.

## 🚀 Bước 1: Cấu hình Cadence trong PyCharm

### 1.1. Mở Cadence Settings

**Windows/Linux:**
```
File → Settings → Tools → Cadence
```

**Mac:**
```
PyCharm → Preferences → Tools → Cadence
```

### 1.2. Đăng nhập JetBrains Account

- Click "Sign in to JetBrains Account"
- Đăng nhập với tài khoản JetBrains của bạn
- Nếu chưa có tài khoản, tạo tại: https://account.jetbrains.com/

### 1.3. Chọn GPU Configuration

Cadence hỗ trợ nhiều loại GPU:

| GPU Type | VRAM | Speed | Cost | Khuyến nghị |
|----------|------|-------|------|-------------|
| T4 | 16GB | ⚡⚡ | $ | Quick experiments |
| A10 | 24GB | ⚡⚡⚡ | $$ | **Recommended** |
| A100 | 40GB | ⚡⚡⚡⚡ | $$$ | Large models |

**Khuyến nghị cho UNet++:**
- **T4**: Đủ cho ResNet50, batch_size=8-16
- **A10**: Tốt nhất cho ResNet50/101, batch_size=16-32
- **A100**: Overkill nhưng rất nhanh

## 🔧 Bước 2: Chuẩn bị Project

### 2.1. Kiểm tra requirements.txt

File `requirements.txt` đã có sẵn với các dependencies cần thiết:

```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.14.0
matplotlib>=3.7.0
numpy>=1.24.0
tqdm>=4.65.0
pandas>=2.0.0
```

### 2.2. Cấu hình dataset path

Sửa `config.py`:

```python
# config.py
DATASET_DFC2020 = "GFM-Bench/DFC2020"  # Cadence sẽ tự download từ HuggingFace
```

Hoặc nếu bạn có dataset local:

```python
DATASET_DFC2020 = "/path/to/your/dfc2020"
```

### 2.3. Điều chỉnh batch size cho GPU

Sửa `config.py` tùy theo GPU:

```python
# Cho T4 (16GB)
STAGE2 = {
    "batch_size": 8,
    "input_size": 96,
    ...
}

# Cho A10 (24GB)
STAGE2 = {
    "batch_size": 16,
    "input_size": 96,
    ...
}

# Cho A100 (40GB)
STAGE2 = {
    "batch_size": 32,
    "input_size": 96,
    ...
}
```

## ▶️ Bước 3: Chạy Training trên Cadence

### 3.1. Sử dụng Run Configurations (Khuyến nghị)

PyCharm đã có sẵn 3 run configurations:

1. **Train UNet++ ResNet50**
   - Backbone: ResNet50
   - Deep supervision: Enabled
   - Visualization: Enabled

2. **Train UNet++ MobileViT**
   - Backbone: MobileViT
   - Deep supervision: Enabled
   - Visualization: Enabled

3. **Compare Results**
   - So sánh kết quả experiments

**Cách chạy:**

1. Click dropdown menu ở toolbar (bên cạnh nút Run)
2. Chọn configuration (ví dụ: "Train UNet++ ResNet50")
3. Click nút **"Run on Cadence"** (icon cloud ☁️)
4. Chọn GPU type và region
5. Click "Start"

### 3.2. Chạy từ Terminal

Mở Terminal trong PyCharm và chạy:

```bash
# ResNet50 + Deep Supervision
python training_unetpp.py --backbone resnet50 --deep_supervision --visualize

# MobileViT + Deep Supervision
python training_unetpp.py --backbone mobilevit --deep_supervision --visualize

# ResNet101 (cần GPU lớn hơn)
python training_unetpp.py --backbone resnet101 --deep_supervision
```

Sau đó click "Run on Cadence" trong terminal.

### 3.3. Chạy Experiments

```bash
# Chạy tất cả 4 experiments
bash run_experiments.sh

# Hoặc từng experiment riêng
python training_unetpp.py --backbone resnet50 --checkpoint_dir experiments/exp1
python training_unetpp.py --backbone resnet50 --deep_supervision --checkpoint_dir experiments/exp2
python training_unetpp.py --backbone mobilevit --deep_supervision --checkpoint_dir experiments/exp3
python training_unetpp.py --backbone resnet101 --deep_supervision --checkpoint_dir experiments/exp4
```

## 📊 Bước 4: Monitor Training

### 4.1. Xem Logs trong PyCharm

- Logs sẽ hiển thị real-time trong PyCharm console
- Bạn sẽ thấy:
  - Training progress (loss, mIoU, accuracy)
  - Epoch time
  - GPU utilization

### 4.2. Checkpoint Auto-sync

Cadence tự động sync checkpoints về local:

```
checkpoints_unetpp_{backbone}_ds/
├── best_model.pth              # Sync về khi có best model mới
├── train_history.csv           # Sync real-time
└── visualizations/             # Sync sau khi training xong
```

### 4.3. Stop/Resume Training

- **Stop**: Click nút Stop trong PyCharm
- **Resume**: Chạy lại với cùng checkpoint_dir, model sẽ tự động resume

## 💰 Bước 5: Quản lý Chi phí

### 5.1. Xem Usage

```
Tools → Cadence → Usage
```

Xem:
- GPU hours used
- Cost estimate
- Remaining credits

### 5.2. Tối ưu chi phí

**Tips:**
1. **Test local trước**: Chạy 1-2 epochs local để đảm bảo code chạy đúng
2. **Dùng GPU phù hợp**: T4 cho experiments, A10 cho final training
3. **Stop khi không dùng**: Đừng để training chạy suông
4. **Batch experiments**: Chạy nhiều experiments cùng lúc để tận dụng GPU

### 5.3. Estimate Cost

| Setup | GPU | Time | Cost (estimate) |
|-------|-----|------|-----------------|
| Quick test (5 epochs) | T4 | ~30 min | ~$0.50 |
| Full training (50 epochs) | T4 | ~5 hours | ~$5.00 |
| Full training (50 epochs) | A10 | ~3 hours | ~$9.00 |
| Full training (50 epochs) | A100 | ~1.5 hours | ~$15.00 |

*Giá chỉ mang tính tham khảo, xem chính xác tại Cadence dashboard*

## 🔍 Bước 6: Xem Kết quả

### 6.1. Download Results

Sau khi training xong, Cadence tự động sync về local:

```bash
# Xem training history
cat checkpoints_unetpp_resnet50_ds/train_history.csv

# Xem visualizations
open checkpoints_unetpp_resnet50_ds/visualizations/
```

### 6.2. Compare Experiments

```bash
python compare_results.py
```

Hoặc dùng run configuration "Compare Results"

### 6.3. Load Best Model

```python
import torch
from src.models.unetplusplus import UNetPlusPlus

# Load model
model = UNetPlusPlus(
    num_classes=8,
    backbone="resnet50",
    input_channels=12,
    input_size=96,
    deep_supervision=True
)

# Load checkpoint
checkpoint = torch.load("checkpoints_unetpp_resnet50_ds/best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")
```

## 🐛 Troubleshooting

### Issue 1: "Cadence not available"

**Giải pháp:**
1. Cập nhật PyCharm lên version mới nhất
2. Kiểm tra JetBrains Account đã đăng nhập
3. Kiểm tra internet connection

### Issue 2: "Out of Memory on GPU"

**Giải pháp:**
```python
# Giảm batch size trong config.py
STAGE2 = {
    "batch_size": 4,  # Giảm từ 8/16
    "input_size": 96,
}
```

Hoặc chọn GPU lớn hơn (A10/A100)

### Issue 3: "Dataset download failed"

**Giải pháp:**
1. Kiểm tra dataset path trong config.py
2. Đảm bảo có internet connection
3. Thử download dataset trước:

```python
from datasets import load_dataset
dataset = load_dataset("GFM-Bench/DFC2020")
```

### Issue 4: "Training too slow"

**Giải pháp:**
1. Bật mixed precision (đã bật mặc định)
2. Tăng batch size nếu có đủ memory
3. Dùng GPU nhanh hơn (A10/A100)
4. Giảm input_size nếu có thể

### Issue 5: "Connection lost"

**Giải pháp:**
- Training vẫn chạy trên cloud
- Reconnect và xem logs
- Checkpoints vẫn được lưu

## 📝 Best Practices

### 1. Development Workflow

```
1. Code local → Test 1-2 epochs local
2. Push to Git (optional)
3. Run on Cadence với full training
4. Download results
5. Analyze và iterate
```

### 2. Experiment Tracking

Tạo file `experiments.md` để track:

```markdown
| Exp | Backbone | Deep Sup | GPU | Time | Best mIoU | Notes |
|-----|----------|----------|-----|------|-----------|-------|
| 1   | resnet50 | No       | T4  | 4h   | 0.6234    | Baseline |
| 2   | resnet50 | Yes      | T4  | 5h   | 0.6789    | +5.5% |
| 3   | mobilevit| Yes      | T4  | 3h   | 0.6456    | Faster |
| 4   | resnet101| Yes      | A10 | 4h   | 0.7012    | Best! |
```

### 3. Checkpoint Management

```bash
# Backup best models
mkdir -p best_models
cp checkpoints_unetpp_resnet50_ds/best_model.pth best_models/resnet50_ds.pth
cp checkpoints_unetpp_resnet101_ds/best_model.pth best_models/resnet101_ds.pth
```

## 🎯 Quick Start Checklist

- [ ] PyCharm installed và updated
- [ ] Cadence configured trong PyCharm
- [ ] JetBrains Account đã đăng nhập
- [ ] requirements.txt đã có
- [ ] config.py đã điều chỉnh batch_size
- [ ] Dataset path đã cấu hình
- [ ] Run configuration đã test
- [ ] GPU type đã chọn
- [ ] Ready to train! 🚀

## 📚 Resources

- **Cadence Documentation**: https://www.jetbrains.com/help/pycharm/cadence.html
- **PyCharm Remote Development**: https://www.jetbrains.com/remote-development/
- **JetBrains Account**: https://account.jetbrains.com/

---

**Happy Training on Cadence! ☁️🚀**

Nếu có vấn đề, check Troubleshooting section hoặc xem PyCharm logs.
