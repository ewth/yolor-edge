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
    IMAGE_SIZE=256
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


SOURCE=0
# SOURCE="/resources/sources/inference.mp4"
# SOURCE="/resources/datasets/darkfacesml/test"
# SOURCE="/resources/datasets/darkfacesml/images"

echo "Starting detection with ${YOLOR_MODEL} at image size ${IMAGE_SIZE}"
echo " on source ${SOURCE} with classes ${CLASS}"
if [[ ! -z "${EXTRA_ARGS}" ]]; then
    echo " and extra args: ${EXTRA_ARGS}"
fi

python /yolor-edge/yolor/detect.py \
    --source ${SOURCE} --conf 0.75 --device 0 \
    --names /yolor-edge/data/coco-2017/coco.names \
    --view-img --display-info --display-bb \
    --output /resources/inference/yolor/output \
    --cfg /yolor-edge/yolor/cfg/${YOLOR_CFG}.cfg \
    --weights /resources/weights/yolor/${YOLOR_MODEL}.pt \
    --class ${CLASS} --img-size ${IMAGE_SIZE}
