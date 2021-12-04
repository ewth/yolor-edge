#!/bin/bash


for BATCH_SIZE in 8 12 20
do
    for IMAGE_SIZE in 256 512
    do
        IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
    done
done



for BATCH_SIZE in 8 12 20 32 50
do
    for IMAGE_SIZE in 128 256 512 1024 1280
    do
        IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} ./run-val.sh
    done
done

sudo systemctl set-default multi-user.target
# MODE_20W_6CORE