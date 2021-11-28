# jetson-yolor

This forms part of a larger project, [DeepRescue](https://github.com/ewth/DeepRescue), examining how artificial intelligence can assist in urban search and rescue situations.
This particular aspect of the project focuses on running YOLOR on an [NVIDIA Jetson Xavier NX](https://developer.nvidia.com/embedded/jetson-xavier-nx-devkit), evaluating its suitability for locating people using computer vision in collapsed building disasters. Being signficantly cluttered, dynamic and unpredictable environments with highly variable lighting, such disaster sites have historically proven difficult for autonomous robotics to operate in.

## YOLOR

The primary object detection algorithm focused on here is You Only Learn One Representation, or YOLOR, published in July 2021.

All credit goes to:
- Chien-Yao Wang, I-Hau Yeh and Hong-Yuan Mark Liao for their groundbreaking paper, [You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/abs/2105.04206).
- Kin-Yiu, Wong for his original implementation of YOLOR in PyTorch, on which this work is built: [YOLOR](https://github.com/WongKinYiu/yolor)

## Torchvision

The author of the [YOLOR implementation](https://github.com/WongKinYiu/yolor) used torchvision 0.8.1 in his work, which isn't readily available in binary form for arm64.
A wheel is included in `container-files` that was built from source on a Jetson Xavier NX.

## Resources

I mount a volume into the container as `/resources` which has in it (amongst other things):

```text

resources/
    |-- datasets/
        |-- cocos2017/          # Cocos 2017 dataset
    |-- weights/
        |-- yolor/
            |-- yolor_p6.pt     # Pretrained YOLOR-P6 weights
            |-- yolor_w6.pt     # Pretrained YOLOR-W6 weights

```

Note that the yolor weights were obtained by running `scripts/get_pretrain.sh` in the YOLOR repo.