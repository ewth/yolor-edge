# jetson-yolor

An attempt at getting YOLOR running on a Jetson Xavier NX.

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