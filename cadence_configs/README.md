# Cadence YAML Configurations

## 📁 Cấu trúc

```
cadence_configs/
├── README.md           # File này
├── resnet50.yaml       # ResNet50 config
├── mobilevit.yaml      # MobileViT config
├── resnet101.yaml      # ResNet101 config
└── quick_test.yaml     # Quick test config
```

## 🚀 Cách sử dụng

### Cách 1: Từ PyCharm UI

1. Mở tab **JetBrains Cadence** (bottom panel)
2. Click **"Run an execution"**
3. Chọn **"Load from YAML"**
4. Browse và chọn file YAML (ví dụ: `cadence_configs/resnet50.yaml`)
5. Review settings và click **"Start"**

### Cách 2: Từ Command Line

```bash
# Sử dụng Cadence CLI (nếu có cài)
cadence run --config cadence_configs/resnet50.yaml

# Hoặc
cadence run cadence_configs/resnet50.yaml
```

### Cách 3: Tự động với .cadence.yaml

File `.cadence.yaml` ở root project sẽ tự động được PyCharm detect.

## 📋 Các config có sẵn

### 1. resnet50.yaml
- **Backbone**: ResNet50
- **GPU**: A10 (24GB)
- **Time**: ~6 hours
- **Cost**: ~$15
- **Use case**: Recommended baseline

```bash
# Chạy từ PyCharm hoặc:
cadence run cadence_configs/resnet50.yaml
```

### 2. mobilevit.yaml
- **Backbone**: MobileViT
- **GPU**: T4 (16GB)
- **Time**: ~4 hours
- **Cost**: ~$8
- **Use case**: Fast experiments, mobile deployment

```bash
cadence run cadence_configs/mobilevit.yaml
```

### 3. resnet101.yaml
- **Backbone**: ResNet101
- **GPU**: A10 (24GB)
- **Time**: ~8 hours
- **Cost**: ~$20
- **Use case**: Best accuracy

```bash
cadence run cadence_configs/resnet101.yaml
```

### 4. quick_test.yaml
- **Backbone**: ResNet50
- **GPU**: T4 (16GB)
- **Time**: ~1 hour
- **Cost**: ~$2
- **Use case**: Quick testing before full training

```bash
cadence run cadence_configs/quick_test.yaml
```

## ✏️ Tùy chỉnh YAML

### Thay đổi GPU

```yaml
resources:
  gpu: T4      # T4 (16GB) - Cheapest
  gpu: A10     # A10 (24GB) - Recommended
  gpu: A100    # A100 (40GB) - Fastest
```

### Thay đổi arguments

```yaml
args:
  backbone: resnet50
  deep_supervision: true
  visualize: true
  encoder_checkpoint: checkpoints_stage1/best_encoder.pth  # Add pretrained encoder
  checkpoint_dir: experiments/exp1  # Custom checkpoint dir
```

### Thay đổi timeout và cost

```yaml
resources:
  timeout: 6h  # Maximum execution time

cost:
  max_cost: 15.00  # Maximum cost in USD
  auto_stop: true  # Auto-stop when done
```

### Thay đổi sync settings

```yaml
sync:
  upload:
    - "*.py"
    - "src/**"
    - "config.py"
    - "my_custom_file.txt"  # Add custom files
  
  download:
    - "checkpoints_*/**"
    - "experiments/**"
    - "*.csv"
    - "*.png"
  
  exclude:
    - "__pycache__"
    - "*.pyc"
    - ".git"
    - "large_file.bin"  # Exclude large files
```

## 🎯 Workflow khuyến nghị

### 1. Quick Test trước
```bash
# Test với 5 epochs để đảm bảo code chạy đúng
cadence run cadence_configs/quick_test.yaml
```

### 2. Baseline với ResNet50
```bash
# Full training với ResNet50
cadence run cadence_configs/resnet50.yaml
```

### 3. Thử các backbone khác
```bash
# Nhanh hơn
cadence run cadence_configs/mobilevit.yaml

# Chính xác hơn
cadence run cadence_configs/resnet101.yaml
```

### 4. So sánh kết quả
```bash
# Sau khi tất cả experiments xong
python compare_results.py
```

## 📊 So sánh configs

| Config | GPU | Time | Cost | Accuracy | Use Case |
|--------|-----|------|------|----------|----------|
| quick_test | T4 | 1h | $2 | - | Testing |
| mobilevit | T4 | 4h | $8 | ⭐⭐⭐ | Fast/Mobile |
| resnet50 | A10 | 6h | $15 | ⭐⭐⭐⭐ | **Recommended** |
| resnet101 | A10 | 8h | $20 | ⭐⭐⭐⭐⭐ | Best |

## 🔧 Troubleshooting

### "YAML file not found"
- Đảm bảo path đúng: `cadence_configs/resnet50.yaml`
- Check working directory

### "Invalid YAML syntax"
- Kiểm tra indentation (dùng spaces, không dùng tabs)
- Validate YAML: https://www.yamllint.com/

### "GPU not available"
- Thử GPU khác (T4 → A10)
- Thử region khác
- Chờ vài phút và thử lại

### "Out of memory"
- Giảm batch_size trong config.py
- Dùng GPU lớn hơn (T4 → A10 → A100)

## 💡 Tips

1. **Test local trước**: Chạy 1-2 epochs local để catch bugs
2. **Start small**: Dùng quick_test.yaml trước
3. **Monitor cost**: Check Cadence dashboard thường xuyên
4. **Backup checkpoints**: Download checkpoints sau mỗi experiment
5. **Use version control**: Commit code trước khi chạy Cadence

## 📚 Resources

- **Cadence Docs**: https://www.jetbrains.com/help/pycharm/cadence.html
- **YAML Syntax**: https://yaml.org/
- **Project README**: ../README.md
- **Full Guide**: ../CADENCE_GUIDE.md

---

**Happy Training! ☁️🚀**
