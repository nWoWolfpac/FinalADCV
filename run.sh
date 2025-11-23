#!/usr/bin/env bash
set -e

# Optional: activate virtual environment if you use one
# source .venv/bin/activate

# Move to project root (directory of this script)
cd "$(dirname "$0")"

# Train Stage 2 segmentation on DFC2020 using SegNet
# Hyperparameters (batch_size, num_epochs, lrs, etc.) are taken from config.py (STAGE2)

python training_decoder.py \
  --model segnet \
  --backbone resnet50 \
  --encoder_checkpoint checkpoints/stage1_encoder.pth \
  --num_classes 19
