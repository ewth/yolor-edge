# yolor-edge

An implementation of [YOLOR](https://github.com/WongKinYiu/yolor) running successfully on the [NVIDIA Jetson Xavier NX](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit) edge computing device.

## YOLOR

[You Only Learn One Representation](https://arxiv.org/abs/2105.04206) (YOLOR) is a novel, state-of-the-art object detection algorithm published in May 2021, and producing [world-leading performance results](https://paperswithcode.com/sota/real-time-object-detection-on-coco).

All credit goes to:
- Chien-Yao Wang, I-Hau Yeh and Hong-Yuan Mark Liao for their groundbreaking paper, [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206).
- Kin-Yiu, Wong for his implementation of [YOLOR in PyTorch](https://github.com/WongKinYiu/yolor) which accompanied the paper above, and on which this work is built.

## This Code

YOLOR was published with an official implementation built on PyTorch ([see below](##YOLOR)), built and tested on a PC. This code represents a replication of the implementation for running on an edge device, with some additions for flair.

## Thesis

This code forms part of a B.Eng(Hons) (Instrumentation and Control / Industrial Computer Systems) thesis, with the aim of evaluating state-of-the-art artificial intelligence research for application in urban search and rescue. As successful implementation and testing of YOLOR on edge devices (embedded systems) was previously unavailable in literature, the overarching aim was assessing suitability for real-world field use in autonomous mobile robotics.

Releases at specific points are available below.

- [Thesis](https://github.com/ewth/yolor-edge/releases/tag/thesis).

## Further Details

The thesis this repository is attached to is slated for publication in February 2022. Further details, including more specifics around implementation and use, will be made available as soon as practical.

## Commit History

Unfortunately, a fair bit of burning the midnight oil and running purely on caffeine led to some kafuffles around commit history in this repo, namely due to tired eyes mishmashing Git submodules.
