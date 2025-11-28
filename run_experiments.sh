#!/bin/bash
# Script to run multiple UNet++ experiments with different configurations

echo "=========================================="
echo "UNet++ Experiments Runner"
echo "=========================================="
echo ""

# Experiment 1: ResNet50 baseline (no deep supervision)
echo ">>> Experiment 1: ResNet50 baseline"
python training_unetpp.py \
    --backbone resnet50 \
    --checkpoint_dir experiments/exp1_resnet50_baseline

echo ""
echo ">>> Experiment 1 completed!"
echo ""

# Experiment 2: ResNet50 with deep supervision
echo ">>> Experiment 2: ResNet50 + Deep Supervision"
python training_unetpp.py \
    --backbone resnet50 \
    --deep_supervision \
    --checkpoint_dir experiments/exp2_resnet50_ds \
    --visualize

echo ""
echo ">>> Experiment 2 completed!"
echo ""

# Experiment 3: MobileViT with deep supervision
echo ">>> Experiment 3: MobileViT + Deep Supervision"
python training_unetpp.py \
    --backbone mobilevit \
    --deep_supervision \
    --checkpoint_dir experiments/exp3_mobilevit_ds \
    --visualize

echo ""
echo ">>> Experiment 3 completed!"
echo ""

# Experiment 4: ResNet101 with deep supervision (best accuracy)
echo ">>> Experiment 4: ResNet101 + Deep Supervision"
python training_unetpp.py \
    --backbone resnet101 \
    --deep_supervision \
    --checkpoint_dir experiments/exp4_resnet101_ds \
    --visualize

echo ""
echo ">>> Experiment 4 completed!"
echo ""

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in experiments/ directory:"
echo "  - exp1_resnet50_baseline/"
echo "  - exp2_resnet50_ds/"
echo "  - exp3_mobilevit_ds/"
echo "  - exp4_resnet101_ds/"
echo ""
echo "Compare results using train_history.csv in each directory"
