ARG UBUNTU_VERSION=16.04
FROM ubuntu:${UBUNTU_VERSION}

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

RUN apt update && apt -y upgrade && apt -y install \
      build-essential \
      cmake \
      git \
      libgtk2.0-dev \
      pkg-config \
      libavcodec-dev \
      libavformat-dev \
      libswscale-dev \
      python-dev \
      python-numpy \
      libtbb2 \
      libtbb-dev \
      libjpeg-dev \
      libpng-dev \
      libtiff-dev \
      libdc1394-22-dev

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} install --upgrade \
    pip \
    setuptools

WORKDIR /app
ADD . /app

RUN ${PIP} install --trusted-host pypi.python.org -r cpu_requirements.txt
