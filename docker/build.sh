#!/usr/bin/env bash

IMAGE_TAG:deep-usar/jetson-yolor:latest
TORCHV_WHL=torchvision-0.8.1-cp36-cp36m-linux_aarch64.whl

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR

if [[ ! -f "Dockerfile" ]]; then
    echo "No Dockerfile found."
    exit
fi


if [ -f ${TORCHV_WHL} ]; then
    echo "Can't find file ${TORCH_WHEEL}"
    exit
fi

sudo docker pull nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3

sudo DOCKER_BUILDKIT=1 docker build \
    --progress plain \
    -t ${IMAGE_TAG} .
