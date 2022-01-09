# yolor-edge

This forms part of a larger project examining how artificial intelligence can assist in urban search and rescue situations.

This particular aspect of the project focuses on implementing and running YOLOR on an [NVIDIA Jetson Xavier NX](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit), evaluating its suitability for locating people using computer vision in collapsed building disasters. Being signficantly cluttered, dynamic and unpredictable environments with highly variable lighting, such disaster sites have historically proven difficult for autonomous robotics to operate in.

## YOLOR

The primary object detection algorithm focused on here is You Only Learn One Representation, or YOLOR, published in May 2021.

All credit goes to:
- Chien-Yao Wang, I-Hau Yeh and Hong-Yuan Mark Liao for their groundbreaking paper, [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206).
- Kin-Yiu, Wong for his original implementation of YOLOR in PyTorch, on which this work is built: [YOLOR](https://github.com/WongKinYiu/yolor)

## Torchvision

The author of the [YOLOR implementation](https://github.com/WongKinYiu/yolor) used torchvision 0.8.1 in his work, which isn't readily available in binary form for aarch64.
A wheel is included in `container-files` that was built from source on a Jetson Xavier NX.

## Resources

A volume is mounted into the container as `/resources` which has in it (amongst other things):

```text

resources/
    |-- datasets/
        |-- coco-2017/          # COCO-2017 dataset
    |-- weights/
        |-- yolor/
            |-- yolor_p6.pt     # Pretrained YOLOR-P6 weights
            |-- yolor_w6.pt     # Pretrained YOLOR-W6 weights

```

Note that the yolor weights were obtained by running `scripts/get_pretrain.sh` in the YOLOR repo.