#!/usr/bin/env bash


SHM_SIZE=4gb
IMAGE=deep-usar/jetson-yolor:version0.1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Bit hacky... better way?
URBASE=$(dirname $(dirname ${SCRIPT_DIR}))
RESOURCES=${URBASE}/resources
JETSON_YOLOR=${URBASE}/jetson-yolor
CONTAINER=${SCRIPT_DIR}/container-files
YOLOR=${URBASE}/yolor
WANDB_DIR=${RESOURCES}/wandb/jetson-yolor

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
    -v ${CONTAINER}:/jetson-yolor \
    -v ${YOLOR}:/yolor \
    --shm-size=${SHM_SIZE} \
    ${V4L2_DEVICES} ${IMAGE}
