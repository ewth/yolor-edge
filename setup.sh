#!/usr/bin/env bash

# I haven't actually run this script per se, but it's how the resources are setup
# Some of this may be unnecessary. But downloading the same (several hundred mb) files over and over is... tedious.

RESOURCES=${HOME}/project/resources

# TorchVision 0.8.1
cd $RESOURCES
mkdir -p software/torchvision
cd software/torchvision
git clone -b v0.8.1 https://github.com/pytorch/vision v0.8.1

# PyTorch 1.7
cd $RESOURCES
mkdir -p wheels
cd wheels
RUN wget https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
