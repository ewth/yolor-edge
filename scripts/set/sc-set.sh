#!/bin/bash


SET_TAG="single_class_pvalA"
cd ../val

# Best mAP
IMAGE_SIZE=1280
BATCH_SIZE=1
for runs in {1..5}
do
	WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./single-class.sh
done

# best perf
IMAGE_SIZE=1024
BATCH_SIZE=1
for runs in {1..5}
do
	WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./single-class.sh
done

# 2nd best perf
IMAGE_SIZE=960
BATCH_SIZE=4
for runs in {1..5}
do
	WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./single-class.sh
done


# 3rd best perf
IMAGE_SIZE=768
BATCH_SIZE=8
for runs in {1..5}
do
	WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./single-class.sh
done

# 4th best perf
IMAGE_SIZE=640
BATCH_SIZE=12
for runs in {1..5}
do
	WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./single-class.sh
done
