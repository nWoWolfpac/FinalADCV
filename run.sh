!/bin/bash
# Script to run multiple UNet++ experiments with different configurations

#echo "=========================================="
#echo "UNet++ Experiments Runner"
#echo "=========================================="
#echo ""
#
## Experiment 1: ResNet50 baseline (no deep supervision)
#echo ">>> Experiment 1: ResNet50 baseline"
#python training_unetpp.py \
#    --backbone resnet50 \
#    --checkpoint_dir experiments/exp1_resnet50_baseline
#
#echo ""
#echo ">>> Experiment 1 completed!"
#echo ""

# Experiment 2: ResNet50 with deep supervision
echo ">>> Experiment 5: mobilenetv4_hybrid"
python training_unetpp.py --backbone mobilenetv4_hybrid --deep_supervision --visualize


echo ""
echo ">>> Experiment 5 completed!"
echo ""