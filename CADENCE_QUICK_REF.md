# Cadence Quick Reference

## 🚀 Chạy nhanh

### Từ PyCharm UI
```
1. Tab "JetBrains Cadence" (bottom)
2. Click "Run an execution"
3. Load YAML: cadence_configs/resnet50.yaml
4. Start
```

### Từ Command Line
```bash
cadence run cadence_configs/resnet50.yaml
```

## 📁 YAML Configs có sẵn

| File | Backbone | GPU | Time | Cost | Use Case |
|------|----------|-----|------|------|----------|
| `quick_test.yaml` | ResNet50 | T4 | 1h | $2 | Test code |
| `mobilevit.yaml` | MobileViT | T4 | 4h | $8 | Fast/Light |
| `resnet50.yaml` | ResNet50 | A10 | 6h | $15 | **Recommended** |
| `resnet101.yaml` | ResNet101 | A10 | 8h | $20 | Best accuracy |

## ⚙️ Tùy chỉnh nhanh

### Thay đổi GPU
```yaml
resources:
  gpu: T4    # 16GB - $1-2/hour
  gpu: A10   # 24GB - $2-3/hour
  gpu: A100  # 40GB - $4-5/hour
```

### Thêm pretrained encoder
```yaml
args:
  backbone: resnet50
  deep_supervision: true
  encoder_checkpoint: checkpoints_stage1/best_encoder.pth
```

### Thay đổi checkpoint directory
```yaml
args:
  backbone: resnet50
  checkpoint_dir: experiments/exp1
```

## 🔍 Validate config

```bash
# Validate tất cả configs
python validate_cadence_config.py

# Validate 1 file cụ thể
python validate_cadence_config.py cadence_configs/resnet50.yaml
```

## 📊 Monitor

### Trong PyCharm
- Tab "JetBrains Cadence" → Xem executions
- Real-time logs
- Auto-sync checkpoints

### Check cost
```
Tools → Cadence → Usage
```

## 🛑 Stop execution

### Từ PyCharm
- Click "Stop" button trong Cadence tab

### Từ CLI
```bash
cadence stop <execution_id>
```

## 📥 Download results

Auto-sync về:
```
checkpoints_unetpp_{backbone}_ds/
├── best_model.pth
├── train_history.csv
└── visualizations/
```

## 💡 Tips

✅ Test local trước (1-2 epochs)  
✅ Dùng quick_test.yaml để test code  
✅ Monitor cost trong Cadence dashboard  
✅ Backup checkpoints sau mỗi experiment  
✅ Commit code trước khi chạy Cadence  

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Giảm batch_size hoặc dùng GPU lớn hơn |
| GPU not available | Thử region khác hoặc GPU khác |
| Script not found | Check working directory |
| YAML syntax error | Validate với `validate_cadence_config.py` |

## 📚 Docs

- Full guide: `CADENCE_GUIDE.md`
- YAML configs: `cadence_configs/README.md`
- Project README: `README.md`

---

**Quick Start:**
```bash
# 1. Validate config
python validate_cadence_config.py cadence_configs/resnet50.yaml

# 2. Run on Cadence (from PyCharm)
# Tab "Cadence" → "Run an execution" → Load resnet50.yaml → Start

# 3. Monitor progress
# Watch logs in PyCharm console

# 4. Compare results
python compare_results.py
```
