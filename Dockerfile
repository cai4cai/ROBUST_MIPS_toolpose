FROM nvcr.io/nvidia/pytorch:21.04-py3
ARG USER_ID
ARG GROUP_ID
ARG USER


ENV FORCE_CUDA="1"
ENV CUDA_HOME="/usr/local/cuda"
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub


RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

# Install xtcocotools
RUN pip install cython
RUN pip install --no-cache-dir xtcocotools==1.12

RUN pip install tensorboard

RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'


# Install MMEngine and MMCV
RUN pip install openmim
RUN mim install mmengine "mmcv>=2.0.0rc4, <2.2.0" "mmdet>=3.1.0" "mmpose>=1.3.1"   
RUN mim install 'mmpretrain>=1.0.0'

WORKDIR /workspace
COPY mmpose/ /workspace/mmpose/
RUN pip install -e /workspace/mmpose

COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt

# COPY surgicaltool_bm/ /workspace/surgicaltool_bm/
COPY surgicaltool_bm/configs/ /workspace/surgicaltool_bm/configs/
COPY surgicaltool_bm/custom_src/ /workspace/surgicaltool_bm/custom_src/
COPY surgicaltool_bm/tools/ /workspace/surgicaltool_bm/tools/
COPY surgicaltool_bm/setup.py /workspace/surgicaltool_bm/

WORKDIR /workspace/surgicaltool_bm
RUN pip install -e .

RUN pip install albumentations
RUN pip uninstall numpy -y
RUN pip install numpy==1.22.4

RUN pip uninstall opencv-python -y && \
    pip uninstall opencv-contrib-python -y && \
    pip uninstall opencv-contrib-python-headless -y &&\
    pip uninstall opencv-python-headless -y  &&\
    pip install opencv-contrib-python==4.5.5.64

EXPOSE 8888