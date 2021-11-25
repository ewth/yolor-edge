#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

TORCHV_WHL=torchvision-0.8.1-cp36-cp36m-linux_aarch64.whl

sudo docker pull nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
sudo DOCKER_BUILDKIT=1 docker build \
    --build-arg TORCHV_WHL=${TORCHV_WHL} \
    --progress plain \
    -t jetson-yolor \
    -f ${SCRIPT_DIR}/Dockerfile .
