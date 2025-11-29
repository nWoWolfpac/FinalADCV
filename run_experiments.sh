#!/bin/bash
# UNet++ Training Script
# Supported backbones: resnet18, resnet50, resnet101, mobilevit

BACKBONE=${1:-resnet50}

echo "=========================================="
echo "UNet++ Training"
echo "Backbone: $BACKBONE"
echo "=========================================="

python training_unetpp.py \
    --backbone $BACKBONE \
    --deep_supervision \
    --visualize

echo "=========================================="
echo "Training completed!"
echo "=========================================="
