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

echo "Starting detection with ${YOLOR_VERSION} at image size ${IMAGE_SIZE}"

# cd /yolor

# python detect.py \
#     --source inference/images/horses.jpg \
#     --cfg cfg/yolor_p6.cfg \
#     --weights yolor_p6.pt
#     --conf 0.25 --img-size 1280 --device 0

# --source /resources/sources/sdvd_v1.avi \

python /yolor/detect.py \
    --source 0 \
    --names /jetson-yolor/data/coco.names \
    --cfg /yolor/cfg/${YOLOR_VERSION}.cfg \
    --weights /resources/weights/yolor/${YOLOR_VERSION}.pt \
    --output /resources/inference/yolor/output \
    --conf 0.25 \
    --class ${CLASS} \
    --img-size ${IMAGE_SIZE} \
    --device 0

