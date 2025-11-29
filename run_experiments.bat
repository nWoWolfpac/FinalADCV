@echo off
REM UNet++ Training Script
REM Supported backbones: resnet18, resnet50, resnet101, mobilevit

set BACKBONE=%1
if "%BACKBONE%"=="" set BACKBONE=resnet50

echo ==========================================
echo UNet++ Training
echo Backbone: %BACKBONE%
echo ==========================================

python training_unetpp.py ^
    --backbone %BACKBONE% ^
    --deep_supervision ^
    --visualize

echo ==========================================
echo Training completed!
echo ==========================================
pause
