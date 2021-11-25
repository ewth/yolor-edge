#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

SHM_SIZE=4gb

# Bit hacky... better way?
URBASE=$(dirname $(dirname ${SCRIPT_DIR}))
RESOURCES=${URBASE}/resources
JETSON_YOLOR=${URBASE}/jetson-yolor
CONTAINER=${SCRIPT_DIR}/container-files
YOLOR=${URBASE}/yolor

V4L2_DEVICES=" "

for i in {0..9}
do
   TARGET_DEV=/dev/video${i}
	if [ -a $TARGET_DEV ]; then
		echo "Adding camera ${TARGET_DEV}"
		V4L2_DEVICES="${V4L2_DEVICES} --device ${TARGET_DEV} "
	fi
done

# Including displays to run from xrdp

sudo xhost +si:localuser:root
sudo docker run --runtime nvidia \
    -it --rm --security-opt  seccomp=unconfined --network host \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /etc/enctune.conf:/etc/enctune.conf \
    -v ${RESOURCES}:/resources \
    -v ${JETSON_YOLOR}:/jetson-yolor \
    -v ${YOLOR}:/yolor \
    --shm-size=${SHM_SIZE} \
    ${V4L2_DEVICES} jetson-yolor:latest
