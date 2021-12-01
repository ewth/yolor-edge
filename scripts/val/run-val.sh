#!/usr/bin/env bash

##
# This script is meant to invoke training with some paramaters attached to allow easy logging (via wandb).
# Mainly just metadata to help with comparing tests etc later.
##

# Thanks https://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [[ -z "${BATCH_SIZE}" ]]; then
    BATCH_SIZE=40
fi
if [[ -z "${IMAGE_SIZE}" ]]; then
    IMAGE_SIZE=128
fi

if [[ -z "${YOLOR_VERSION}" ]]; then
    YOLOR_VERSION=yolor_p6
fi

echo "Starting validation of ${YOLOR_VERSION} with batch size ${BATCH_SIZE}, image size ${IMAGE_SIZE}"

# This is just read to pass to wandb
SHM_SIZE=$(cat /proc/mounts | grep "/dev/shm" | grep -Po 'size=([0-9]+[a-zA-Z])' | grep -Po '([0-9]+[a-zA-Z])')
if [[ ! $SHM_SIZE =~ [0-9]+[a-zA-Z] ]]; then
    SHM_SIZE=""
else
    SHM_FACT=${SHM_SIZE: -1}
    SHM_DIV=1
    SHM_ADD=""
    if [ "${SHM_FACT,,}" == "k" ]; then
        # kbytes
        SHM_DIV=1024
    fi
    SHM_SIZE=$((${SHM_SIZE::-1} / ${SHM_DIV}))
fi

SHM_SIZE=${SHM_SIZE} python3 /yolor-edge/yolor/test.py \
    --verbose \
    --data /yolor-edge/data/coco.yaml \
    --names /yolor-edge/data/coco.names \
    --img-sz ${IMAGE_SIZE} --batch ${BATCH_SIZE} --conf 0.001 --iou 0.65 --device 0 \
    --cfg /yolor-edge/yolor/cfg/${YOLOR_VERSION}.cfg \
    --weights /resources/weights/yolor/${YOLOR_VERSION}.pt \
    --project /resources/runs/yolor/test \
    --name ${YOLOR_VERSION}_val