#!/bin/bash

# Now we want to explore:
#	Infsize 512: Batch size 30, 31, 32, 33
#	Infsize 128 and 256: Batch size 30, 32, 34, 40, 50
#	Infsize 960: 2, 4, 6

SET_TAG="init_perf_eval_Batch_E"

for BATCH_SIZE in 30 31 32 33 34
do
	IMAGE_SIZE=512
	WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
done


for BATCH_SIZE in 30 32 34 40 50
do
	for IMAGE_SIZE in 128 256
	do
		WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
	done
done

for BATCH_SIZE in 2 4 6 8
do
	IMAGE_SIZE=960
	WANDB_TAGS=${SET_TAG} IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
done


