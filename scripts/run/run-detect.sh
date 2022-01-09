#!/usr/bin/env bash

##
# Run detection
##

# Thanks https://stackoverflow.com/questions/59895/
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [[ -z "${YOLOR_VERSION}" ]]; then
    YOLOR_VERSION=yolor_p6
fi

# Screen size must be 4:3.
# Valid widths include: 1280, 1024, 768, 512, 256

if [[ -z "${IMAGE_SIZE}" ]]; then
    IMAGE_SIZE=1280
fi

if [[ -z "${CLASS}" ]]; then
    # 0 = person
    CLASS=0
fi

if [[ -z "${EXTRA_ARGS}" ]]; then
    EXTRA_ARGS=""
fi

if [[ ! -z "${VERBOSE}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --verbose"
fi

if [[ ! -z "${NTH_FRAME}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --nth-frame ${NTH_FRAME}"
fi

if [[ -z "${YOLOR_CFG}" ]]; then
    YOLOR_CFG=yolor_p6
fi
if [[ -z "${YOLOR_MODEL}" ]]; then
    YOLOR_MODEL=${YOLOR_CFG}
fi
if [[ -z "${PROJECT_NAME}" ]]; then
    PROJECT_NAME="${YOLOR_MODEL}_val"
fi


# SOURCE=0
# SOURCE="/resources/sources/inference.mp4"
# SOURCE="/resources/datasets/darkfacesml/test"
# SOURCE="/resources/datasets/darkfacesml/images"
# SOURCE="/resources/sources/videos/third_test/pexels-2777822.mp4"
SOURCE="/resources/sources/videos/test_6"
OUT_DIR="/resources/inference/yolor/output/test_6/text_scaling"

#yolor_p6 yolor-p6-paper-541
# yolor-w6-paper-555 yolor_w6

IMAGE_SIZE=256

# for IMAGE_SIZE in 1280 512 256
# do
    YOLOR_MODEL=yolor_p6
    YOLOR_CFG=yolor_p6

    echo "Starting detection with ${YOLOR_MODEL} at image size ${IMAGE_SIZE}"
    if [[ ! -z "${EXTRA_ARGS}" ]]; then
        echo " and extra args: ${EXTRA_ARGS}"
    fi
    PASS_OUT_PATH=${OUT_DIR}/${YOLOR_MODEL}_${IMAGE_SIZE}
    mkdir -p ${PASS_OUT_PATH}
    chmod u+rwx ${PASS_OUT_PATH}
    python /yolor-edge/yolor/detect.py \
        --source ${SOURCE} --conf 0.5 --device 0 \
        --names /yolor-edge/data/coco-2017/coco.names \
        --display-info --display-bb --verbose \
        --output ${OUT_DIR}/${YOLOR_MODEL}_${IMAGE_SIZE} \
        --cfg /yolor-edge/yolor/cfg/${YOLOR_CFG}.cfg \
        --weights /resources/weights/yolor/${YOLOR_MODEL}.pt \
        --save-frames --nth-frame 20 \
        --img-size ${IMAGE_SIZE}

    chmod u+rwx ${PASS_OUT_PATH}/*
    YOLOR_MODEL=yolor_w6
    YOLOR_CFG=yolor_w6

#     echo "Starting detection with ${YOLOR_MODEL} at image size ${IMAGE_SIZE}"
#     if [[ ! -z "${EXTRA_ARGS}" ]]; then
#         echo " and extra args: ${EXTRA_ARGS}"
#     fi
#     mkdir -p ${OUT_DIR}/${YOLOR_MODEL}_${IMAGE_SIZE}
#     python /yolor-edge/yolor/detect.py \
#         --source ${SOURCE} --conf 0.5 --device 0 \
#         --names /yolor-edge/data/coco-2017/coco.names \
#         --display-info --display-bb --verbose \
#         --output ${OUT_DIR}/${YOLOR_MODEL}_${IMAGE_SIZE} \
#         --cfg /yolor-edge/yolor/cfg/${YOLOR_CFG}.cfg \
#         --weights /resources/weights/yolor/${YOLOR_MODEL}.pt \
#         --img-size ${IMAGE_SIZE}
# done