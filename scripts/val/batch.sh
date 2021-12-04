#!/bin/bash

# We want to explore:
#   Inference between 512 and 1024: 576 640 704 768 832 896 960
#   Batch sizes above 20 at 512
#   Batch sizes below 8 at 1024

for BATCH_SIZE in 4 8 12 20
do
    for IMAGE_SIZE in 640 768 960
    do
        IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
    done
done


for BATCH_SIZE in 24 30 40 60 80 100
do
    for IMAGE_SIZE in 512
    do
        IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
    done
done


for BATCH_SIZE in 1 2 4 6
do
    for IMAGE_SIZE in 960 1024 1280
    do
        IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
    done
done