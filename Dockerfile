# ------------------------------------------------------------------
# CUDA / CUDNN  10.1 / 7.
# python        3.6    (apt)
# pytorch       1.1.0  (pip)
# opencv        3.4.5  (git)
# ==================================================================

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

# ==================================================================
# ENVIRONMENT VARIABLES
# ------------------------------------------------------------------

ENV LANG C.UTF-8
ENV DATASET_PATH=/home/FasterSeg/dataset
ENV TENSORRT=TensorRT-5.1.5.0.Ubuntu-16.04.5.x86_64-gnu.cuda-10.1.cudnn7.5.tar.gz

# ==================================================================
# TOOLS
# ------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        libsm6 \
        libxext6 \
        libxrender-dev

RUN git clone --depth 10 https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install

# ==================================================================
# PYTHON
# ------------------------------------------------------------------

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py

RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python

RUN DEBIAN_FRONTEND=noninteractive python -m pip --no-cache-dir install --upgrade \
        setuptools \
        requests \
        easydict==1.9 \
        protobuf==3.8.0 \
        numpy==1.16.1 \
        pandas==0.22.0 \
        Pillow==6.2.0 \
        python-dateutil==2.7.3

RUN DEBIAN_FRONTEND=noninteractive python -m pip --no-cache-dir install --upgrade \
        scipy==1.1.0 \
        onnx==1.5.0 \
        tqdm==4.25.0 \
        scikit-learn \
        Cython \
        opencv-python==3.4.5.20 \
        matplotlib==3.0.0

# ==================================================================
# CITYSCAPES SCRIPTS
# ------------------------------------------------------------------

RUN git clone --depth 10 https://github.com/mcordts/cityscapesScripts ~/cityscapesScripts && \
    cd ~/cityscapesScripts && pip install .

# ==================================================================
# OPENCV
# ------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler

RUN git clone --depth 10 --branch 3.4.5 https://github.com/opencv/opencv ~/opencv && \
    mkdir -p ~/opencv/build && cd ~/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_IPP=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_OPENCL=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          .. && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2


# ==================================================================
# PYTORCH
# ------------------------------------------------------------------

RUN cd /tmp && \
    curl -O "https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl"  && \
    curl -O "https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl" && \
    python -m pip --no-cache-dir install \
        /tmp/torch-1.1.0-cp36-cp36m-linux_x86_64.whl \
        /tmp/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl


RUN python -m pip --no-cache-dir install --upgrade  \
        tensorflow==1.9.0 \
        tensorboardX==1.6 \
        thop

# ==================================================================
# PYCUDA
# https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-515/tensorrt-install-guide/index.html#installing-pycuda
# ------------------------------------------------------------------

RUN python -m pip --no-cache-dir install \
        'pycuda>=2017.1.1'

# ==================================================================
# TENSOR-RT v5.1.5.0
# https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-515/tensorrt-install-guide/index.html
# ------------------------------------------------------------------

#COPY /TensorRT-5.1.5.0.Ubuntu-16.04.5.x86_64-gnu.cuda-10.1.cudnn7.5.tar.gz /
COPY /$TENSORRT /
RUN tar -xzvf $TENSORRT

RUN echo 'export LD_LIBRARY_PATH=/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc" && \
    cd /TensorRT-5.1.5.0/python && \
    python -m pip install tensorrt-5.1.5.0-cp36-none-linux_x86_64.whl

# ==================================================================
# DOWNLOAD FASTERSEG-REPOSITORY && INSTALL REQUIREMENTS
# ------------------------------------------------------------------

RUN cd /home/ && git clone https://github.com/Gaussianer/FasterSeg.git
#RUN cd /home/FasterSeg/ && pip install -r requirements.txt

# ==================================================================
# CONFIG & CLEANUP
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*

WORKDIR /home/FasterSeg

EXPOSE 6006 6006
