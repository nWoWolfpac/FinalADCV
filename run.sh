#!/usr/bin/env bash
set -e

#cd "$(dirname "$0")"
echo  "=================== Starting Training ================="

python training_decoder.py \
  --model segnet \
  --backbone resnet50 \
  --num_classes 19

echo  "=================== END Training ================="