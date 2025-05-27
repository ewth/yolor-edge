# yolor-edge

An implementation of [YOLOR](https://github.com/WongKinYiu/yolor) running successfully on the [NVIDIA Jetson Xavier NX](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit) edge computing device.

## Overview

`yolor-edge` was part of an engineering Honours Thesis.

The aim was to utilise state-of-the-art object detection on off-the-shelf hardware for assisting in urban search and rescue.

At the time (2021), YOLOR was one of the state-of-the-art real-time object detection models, achieving 55.4 mAP / 73.3 AP50 / 60.6 AP75 on the COCO dataset. See [COCO test-dev Benchmark (Object Detection) | Papers with Code](https://paperswithcode.com/sota/object-detection-on-coco) for more details.

Here, YOLOR is successfully implemented to run on an edge device, achieving real-time object detection.

## Usage

The following occurs on the NVIDIA Jetson Xavier NX.

```shell
# Clone the repo
git clone https://github.com/ewth/yolor-edge.git

# Change into the repo directory
cd yolor-edge

# Build the Docker image
cd docker
./build.sh

# Run the Docker container
./run.sh
```

Any time you want to run the Docker container, execute `run.sh` from the `docker` directory.

From within the Docker container:

```shell
# Run yolor-edge
python3 yoloredge.py
```

### Bash Scripts

If the Bash scripts will not run, try adding the `+x` permission:

```shell
chmod +x ./run.sh
```

Alternatively, use `bash` to execute directly:

```shell
bash run.sh
```

## YOLOR

[You Only Learn One Representation](https://arxiv.org/abs/2105.04206) (YOLOR) is a novel, state-of-the-art object detection algorithm published in May 2021, and producing [world-leading performance results](https://paperswithcode.com/sota/real-time-object-detection-on-coco).

YOLOR was published with an official implementation built on PyTorch, built and tested on a PC. The code was originally forked from: [YOLOR in PyTorch](https://github.com/WongKinYiu/yolor).
