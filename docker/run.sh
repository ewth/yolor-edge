#!/usr/bin/env bash


SHM_SIZE=8gb
IMAGE=ewth/yolor-edge:latest

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Bit hacky... better way?
DRBASE=$(dirname $(dirname ${SCRIPT_DIR}))
RESOURCES=${DRBASE}/resources
YOLOR_EDGE=${DRBASE}/yolor-edge
CONTAINER=${SCRIPT_DIR}/container-files
WANDB_DIR=${RESOURCES}/wandb/yolor-edge

# Cheers Dusty!
V4L2_DEVICES=""
for i in {0..9}
do
   TARGET_DEV=/dev/video${i}
	if [ -a $TARGET_DEV ]; then
		echo "Adding camera ${TARGET_DEV}"
		V4L2_DEVICES="${V4L2_DEVICES} --device ${TARGET_DEV} "
	fi
done

# Including displays to run from xrdp
ADD_DISPLAY=""
if [[ ! -z "${DISPLAY}" ]]; then
   sudo xhost +si:localuser:root
   ADD_DISPLAY="-e DISPLAY=${DISPLAY}"
fi

ENV_ARGS=""
if [[ ! -z "${WANDB_API_KEY}" ]]; then
    ENV_ARGS="${ENV_ARGS} -e WANDB_API_KEY=""${WANDB_API_KEY}"""
fi
if [[ ! -z "${WANDB_PROJECT}" ]]; then
    ENV_ARGS="${ENV_ARGS} -e WANDB_PROJECT=""${WANDB_PROJECT}"""
fi
if [[ ! -z "${WANDB_ENTITY}" ]]; then
    ENV_ARGS="${ENV_ARGS} -e WANDB_ENTITY=""${WANDB_ENTITY}"""
fi
if [[ ! -z "${WANDB_SILENT}" ]]; then
    ENV_ARGS="${ENV_ARGS} -e WANDB_SILENT=""${WANDB_SILENT}"""
fi

sudo docker run --runtime nvidia \
    -it --rm --security-opt  seccomp=unconfined --network host ${ADD_DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /etc/enctune.conf:/etc/enctune.conf \
    -v ${WANDB_DIR}:/wandb \
    -e WANDB_DIR="/wandb" ${ENV_ARGS} \
    -v ${RESOURCES}:/resources \
    -v ${YOLOR_EDGE}:/yolor-edge \
    -v ${DRBASE}/.git/modules/yolor-edge:/.git/modules/yolor-edge \
    --shm-size=${SHM_SIZE} \
    ${V4L2_DEVICES} ${IMAGE}
