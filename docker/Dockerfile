# syntax = docker/dockerfile:1.3
FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
# Setup files
ARG TORCHV_WHL=torchvision-0.8.1-cp36-cp36m-linux_aarch64.whl
COPY container-files jetson-yolor

# Prevent docker from caching apt, let buildkit handle it
RUN rm -f /etc/apt/apt.conf.d/docker-clean
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get purge -y python2.7 \
    && apt-get update \
    && apt-get install -y \
        zip \
        htop \
        libgl1-mesa-glx \
        build-essential \
            libopenblas-base \
            libopenmpi-dev 

# Install specific pytorch/torchvision versions
# Thanks: https://github.com/dusty-nv/jetson-containers/blob/master/Dockerfile.pytorch
# RUN cd /jetson-yolor \
#     && bash torch.sh

RUN --mount=type=cache,target=/root/.cache/pip \
    rm -f /usr/bin/pip /usr/bin/python \
    && cp /usr/bin/pip3 /usr/bin/pip \
    && cp /usr/bin/python3 /usr/bin/python \
    && pip3 install --upgrade-strategy eager --upgrade setuptools pip \
    && pip3 install Cython \
    && pip3 install numpy /jetson-yolor/tmp/${TORCHV_WHL} \
    && pip3 install -r /jetson-yolor/requirements.txt --disable-pip-version-check \
    && rm -rf /jetson-yolor \
    && echo "Build time: $(date)" > /build-time

# Note: add the following to RUN to install mish-cuda:
#   && cd / && git clone https://github.com/JunnYu/mish-cuda && cd mish-cuda && python3 setup.py build install && cd / \

# RUN git clone https://github.com/WongKinYiu/yolor /yolor \
#     && rm -f /yolor/test.py \ 
#     && rm -f /yolor/requirements.txt \
#     && ln -s /jetson-yolor/yolor/test.py /yolor/test.py \
#     && ln -s /jetson-yolor/yolor/requirements.txt /yolor/requirements.txt
