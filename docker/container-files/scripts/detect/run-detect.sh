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
    IMAGE_SIZE=512
fi

if [[ -z "${CLASS}" ]]; then
    # 0 = person
    CLASS=0
fi

echo "Starting detection with ${YOLOR_VERSION} at image size ${IMAGE_SIZE}"


python /yolor/detect.py \
    --source 0 --conf 0.75 --device 0 \
    --names /jetson-yolor/data/coco.names \
    --output /resources/inference/yolor/output \
    --cfg /yolor/cfg/${YOLOR_VERSION}.cfg \
    --weights /resources/weights/yolor/${YOLOR_VERSION}.pt \
    --class ${CLASS} --img-size ${IMAGE_SIZE}
