#!/usr/bin/env bash

##
# This script is meant to invoke training with some paramaters attached to allow easy logging (via wandb).
# Mainly just metadata to help with comparing tests etc later.
##

# Thanks https://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [[ -z "${BATCH_SIZE}" ]]; then
    BATCH_SIZE=20
fi
if [[ -z "${IMAGE_SIZE}" ]]; then
    IMAGE_SIZE=256
fi

if [[ -z "${YOLOR_VERSION}" ]]; then
    YOLOR_VERSION=yolor_p6
fi

echo "Starting validation of ${YOLOR_VERSION} with batch size ${BATCH_SIZE}, image size ${IMAGE_SIZE}"

# This is just read to pass to wandb
EXTRA_CMD=""
READ_SHM=$(cat /proc/mounts | grep "/dev/shm")
READ_SHM=$(echo "$READ_SHM" | grep -P 'size=([0-9]+[a-zA-Z])' -o)
SHM_SIZE=$(echo "$READ_SHM" | grep -P '([0-9]+[a-zA-Z])' -o)
if [[ !$SHM_SIZE =~ [0-9]+[a-zA-Z] ]]; then
    SHM_SIZE=""
fi


SHM_SIZE=${SHM_SIZE} python3 /yolor/test.py \
    --data /jetson-yolor/data/coco.yaml \
    --names /jetson-yolor/data/coco.names \
    --img ${IMAGE_SIZE} --batch ${BATCH_SIZE} --conf 0.001 --iou 0.65 --device 0 \
    --cfg /yolor/cfg/${YOLOR_VERSION}.cfg \
    --weights /resources/weights/yolor/${YOLOR_VERSION}.pt \
    --project /resources/runs/yolor/test \
    --name ${YOLOR_VERSION}_val