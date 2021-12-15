#!/bin/bash

IMAGE_SIZE=1280
BATCH_SIZE=1
YOLOR_MODEL=yolor_p6
YOLOR_CFG=yolor_p6

# The test.py script reverts to these defaults too
TEST_NAMES="/yolor-edge/data/pascal-2021/pascal.names"
TEST_DATA="/yolor-edge/data/pascal-2021/pascal.yaml"
DATASET="pascal-2012"

PROJECT_NAME="pascal_${YOLOR_MODEL}_val"

source ../run-val.sh

