# Copyright (c) 2023 Aida DSP (aidadsp.cc)
#
# Source code courtesy of Massimo Pennazio <maxipenna@libero.it>
#
# Example usage
# build: docker build --build-arg host_uid=$UID --build-arg host_gid=$(id -g) . -t pytorch
# run: docker run --gpus all -v $PWD:/workdir:rw -w /workdir -p 8888:8888 --env JUPYTER_TOKEN=aidadsp -it pytorch:latest

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer "Massimo Pennazio <maxipenna@libero.it>"

WORKDIR /workdir

RUN pip3 --disable-pip-version-check --no-cache-dir install jupyterlab==3.6.1 librosa==0.10.0 auraloss==0.4.0 \
    tensorflow==2.11.0 tensorboard==2.11.0  \
    noisereduce==2.0.1 \
    pydantic==2.7.3

ENV USER_NAME aidadsp

# Create a user with same uid and gid of the current host's user
ARG host_uid=1000
ARG host_gid=1000
RUN groupadd -g $host_gid $USER_NAME && \
    useradd -g $host_gid -m -s /bin/bash -u $host_uid $USER_NAME

# From this line below all commands run as user
USER $USER_NAME

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
