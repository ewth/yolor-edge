#!/usr/bin/env bash

##
# This script is meant to invoke training with some paramaters attached to allow easy logging (via wandb).
# Mainly just metadata to help with comparing tests etc later.
# Should only be run inside Docker!
##

# Thanks https://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

SINGLE_CLS=""

if [[ ! -z "${DEEPRESCUE}" ]]; then
    YOLOR_NAMES="/yolor-edge/data/deeprescue/coco.names"
    YOLOR_DATA="/yolor-edge/data/deeprescue/coco.yaml"
    SINGLE_CLS="--single-class"
fi

if [[ ! -z "${BATCH_SIZE}" ]]; then
    BATCH_SIZE="--batch-size ${BATCH_SIZE}"
fi
if [[ ! -z "${IMAGE_SIZE}" ]]; then
    IMAGE_SIZE="--img-sz ${IMAGE_SIZE}"
fi
if [[ ! -z "${YOLOR_DATA}" ]]; then
    YOLOR_DATA="--data ${YOLOR_DATA}"
fi
if [[ ! -z "${YOLOR_NAMES}" ]]; then
    YOLOR_NAMES="--names ${YOLOR_NAMES}"
fi
if [[ -z "${YOLOR_VERSION}" ]]; then
    YOLOR_VERSION=yolor_p6
fi


if [[ ! -z "${DEEPRESCUE}" ]]; then
    echo "Starting validation of ${YOLOR_VERSION} on deeprescue data"
else
    echo "Starting validation of ${YOLOR_VERSION} on COCO-2017 data"
fi

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
    ${IMAGE_SIZE} ${BATCH_SIZE} ${YOLOR_DATA} ${YOLOR_NAMES} ${SINGLE_CLS} --verbose \
    --is-coco \
    --cfg /yolor-edge/yolor/cfg/${YOLOR_VERSION}.cfg \
    --weights /resources/weights/yolor/${YOLOR_VERSION}.pt \
    --name ${YOLOR_VERSION}_val