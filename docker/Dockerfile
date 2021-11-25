# syntax = docker/dockerfile:1.3
FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
MAINTAINER Ewan Thompson <ewanpt@gmail.com>
# Setup files
COPY container-files jetson-yolor
ARG WANDB_API_KEY=""
ARG WANDB_PROJECT=""
ARG WANDB_ENTITY=""
ARG WANDB_SILENT=""

ENV WANDB_API_KEY ${WANDB_API_KEY}
ENV WANDB_PROJECT ${WANDB_PROJECT}
ENV WANDB_ENTITY ${WANDB_ENTITY}
ENV WANDB_SILENT=${WANDB_SILENT}

RUN --mount=type=cache,target=/root/.cache/pip \
    rm -f /usr/bin/pip /usr/bin/python \
    && cp /usr/bin/pip3 /usr/bin/pip \
    && cp /usr/bin/python3 /usr/bin/python \
    && pip3 install --upgrade-strategy eager --upgrade setuptools pip \
    && pip3 install -r /jetson-yolor/requirements.txt --disable-pip-version-check \
    && pip3 install /jetson-yolor/torchvision-0.8.1-cp36-cp36m-linux_aarch64.whl \
    && rm -rf /jetson-yolor \
    && echo "Build time: $(date)" > /build-time

# Note: add the following to RUN to install mish-cuda:
#   && cd / && git clone https://github.com/JunnYu/mish-cuda && cd mish-cuda && python3 setup.py build install && cd / \

# RUN git clone https://github.com/WongKinYiu/yolor /yolor \
#     && rm -f /yolor/test.py \ 
#     && rm -f /yolor/requirements.txt \
#     && ln -s /jetson-yolor/yolor/test.py /yolor/test.py \
#     && ln -s /jetson-yolor/yolor/requirements.txt /yolor/requirements.txt
