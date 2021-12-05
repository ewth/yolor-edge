#!/bin/bash

# Now we want to explore:
#	Infsize 768: 14, 16, 18
#	Infsize 640: 24, 26, 32, 34, 36, 38
#	Infsize 512: 32, 34, 36, 38

BATCH_TAG="init_perf_eval_Batch_D"

for BATCH_SIZE in 14 16 18
do
	IMAGE_SIZE=768
	WANDB_TAGS=${BATCH_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
done


for BATCH_SIZE in 24 26 32 34 36 38
do
	IMAGE_SIZE=640
	WANDB_TAGS=${BATCH_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
done

for BATCH_SIZE in 32 34 36 38
do
	IMAGE_SIZE=512
	WANDB_TAGS=${BATCH_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
done


