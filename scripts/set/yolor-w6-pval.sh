#!/bin/bash

# Conclusion:
# Best mAP: 1280 at bsz 1, 0.525126601	0.707319024
# Best mAP/runtime tradeoff at 1024x1024, batch size 4
# 2nd: 960, 4
# 3rd: 768, 8
# 4th: 640, 12

SET_TAG="test_w6"
cd ../val
YOLOR_CFG=yolor_w6

# Best mAP
IMAGE_SIZE=1280
BATCH_SIZE=1
for runs in {1..5}
do
	YOLOR_CFG=${YOLOR_CFG} WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
done

# best perf
IMAGE_SIZE=1024
BATCH_SIZE=4
for runs in {1..5}
do
	YOLOR_CFG=${YOLOR_CFG} WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
done

# 2nd best perf
IMAGE_SIZE=960
BATCH_SIZE=4
for runs in {1..5}
do
	YOLOR_CFG=${YOLOR_CFG} WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
done


# 3rd best perf
IMAGE_SIZE=768
BATCH_SIZE=8
for runs in {1..5}
do
	YOLOR_CFG=${YOLOR_CFG} WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
done

# 4th best perf
IMAGE_SIZE=640
BATCH_SIZE=12
for runs in {1..5}
do
	YOLOR_CFG=${YOLOR_CFG} WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
done
