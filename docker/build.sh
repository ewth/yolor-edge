#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
if [[ "${SCRIPT_DIR}" != $(echo pwd) ]]; then
    echo "Please run the script from the directory ${SCRIPT_DIR}"
    exit
fi
if [[ ! -f "${SCRIPT_DIR}/Dockerfile" ]]; then
    echo "No Dockerfile found."
    exit
fi

TORCHV_WHL=torchvision-0.8.1-cp36-cp36m-linux_aarch64.whl

cd ${SCRIPT_DIR}
if [ -f ${TORCHV_WHL} ]; then
    echo "Can't find file ${TORCH_WHEEL}"
    exit
fi

sudo docker pull nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
sudo DOCKER_BUILDKIT=1 docker build \
    --build-arg TORCHV_WHL=${TORCHV_WHL} \
    --progress plain \
    -t jetson-yolor
