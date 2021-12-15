# syntax = docker/dockerfile:1.3
FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
MAINTAINER Ewan Thompson <contact@ewan.me>
COPY container-files yolor-edge
RUN --mount=type=cache,target=/root/.cache/pip \
    rm -f /usr/bin/pip /usr/bin/python \
    && cp /usr/bin/pip3 /usr/bin/pip \
    && cp /usr/bin/python3 /usr/bin/python \
    && pip3 install --upgrade-strategy eager --upgrade setuptools pip \
    && pip3 install -r /yolor-edge/requirements.txt --disable-pip-version-check \
    && pip3 install /yolor-edge/torchvision-0.8.1-cp36-cp36m-linux_aarch64.whl \
    && rm -rf /yolor-edge \
    && echo "Build time: $(date)" > /build-time
