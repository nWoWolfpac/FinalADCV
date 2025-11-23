#!/usr/bin/env bash
set -e

# cd "$(dirname "$0")"

# List of backbones to test
BACKBONES=("resnet18" "resnet50" "resnet101")

MODEL="segnet"   
NUM_CLASSES=19

for BB in "${BACKBONES[@]}"; do
  echo "=============================="
  echo "Running model=${MODEL} with backbone=${BB}"
  echo "=============================="

  EXP_DIR="checkpoints_stage2_${MODEL}_${BB}"
  mkdir -p "${EXP_DIR}"

  python training_decoder.py \
    --model "${MODEL}" \
    --backbone "${BB}" \
    --num_classes "${NUM_CLASSES}" \
    --encoder_checkpoint "" \
    > "${EXP_DIR}/train.log" 2>&1

  echo "Finished model=${MODEL}, backbone=${BB}. Logs and checkpoints in ${EXP_DIR}"
  echo

done
