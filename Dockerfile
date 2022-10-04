FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu18.04

WORKDIR /ml

# These should address the hopefully transient issues related to nvidia rotating the GPG keys.
# https://forums.developer.nvidia.com/t/invalid-public-key-for-cuda-apt-repository/212901/11
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 11
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub 2

RUN apt-get update && \
    apt-get install -y curl \
        ffmpeg \
        git \
        libbz2-dev \
        libffi-dev \
        libgl1-mesa-glx \
        libglu1 \
        libglfw3 \
        liblzma-dev \
        libosmesa6-dev \
        libsm6 \
        libsqlite3-dev \
        libssl-dev \
        libxext6 \
        patchelf \
        python-dev \
        python3-setuptools \
        vim \
        wget \
        zlib1g-dev

RUN cd /opt && \
    wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz && \
    tar -xvf Python-3.9.7.tgz && \
    (cd Python-3.9.7 && \
      ./configure --enable-optimizations --enable-shared --enable-loadable-sqlite-extensions && \
      make altinstall && \
      ln -s /usr/local/include/python3.9m /usr/local/include/python3.9 && \
      ldconfig) && \
    rm -rf Python-3.9.7.tgz Python-3.9.7

RUN alias python='/usr/local/bin/python3.9'

RUN alias pip3.9="python3.9 -m pip"

RUN pip3.9 install --upgrade pip==20.1.1 setuptools==45.3.0

RUN cd /opt && \
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz && \
    tar -xvf mujoco210-linux-x86_64.tar.gz && \
    rm -rf mujoco210-linux-x86_64.tar.gz

ENV MUJOCO_PY_MUJOCO_PATH="/opt/mujoco210"

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MUJOCO_PY_MUJOCO_PATH}/bin"

ENV MUJOCO_GL=egl

RUN ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

COPY requirements.txt .

RUN pip3.9 install -r requirements.txt

RUN cd /opt && \
    git clone https://github.com/cocodataset/cocoapi && \
    cd cocoapi/PythonAPI && \
    sed -i "s/python/python3.9/g" Makefile && \
    make && \
    cd /opt && \
    rm -rf cocoapi

ENV PYTHONPATH="/ml"
