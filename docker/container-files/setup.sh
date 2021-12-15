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


## Building

## Load a Docker container
sudo docker run --runtime nvidia \
    -it --rm --security-opt  seccomp=unconfined --network host \
		-v $HOME/project/container-mount:/container-mount \
		nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

## Within Docker
apt-get update && apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
export PYTORCH_VERSION=1.7.0
export BUILD_VERSION=0.8.1
git clone --branch v${BUILD_VERSION} https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install --user
cd ../
pip3 wheel .
# Result: torchvision-0.8.1-cp36-cp36m-linux_aarch64.whl
cp torchvision-0.8.1-cp36-cp36m-linux_aarch64.whl /project/container-mount


https://github.com/dusty-nv/jetson-inference/commit/bbabdd7b97b80f7577c576086ef41c953fc6aa59
--security-opt seccomp=unconfined