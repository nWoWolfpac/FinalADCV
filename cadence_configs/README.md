# UNet++ Cadence Config

## Supported Backbones

- `resnet18` - Fastest
- `resnet50` - Recommended
- `resnet101` - Best accuracy


## Cách sử dụng

### 1. Sửa backbone trong `unetpp.yaml`

```yaml
args:
  backbone: resnet50  # Đổi thành backbone bạn muốn
```

### 2. Load config trong Cadence

1. Tab "JetBrains Cadence"
2. Click "Run an execution"
3. Load file: `cadence_configs/unetpp.yaml`
4. Start

## Hoặc chạy từ command line

```bash
# Linux/Mac
bash run_experiments.sh resnet50

# Windows
run_experiments.bat resnet50
```

Thay `resnet50` bằng backbone bạn muốn.
