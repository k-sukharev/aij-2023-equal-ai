# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
FROM cr.msk.sbercloud.ru/aicloud-base-images/cuda11.8-torch2-py310
USER root

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
# RUN DS_BUILD_FUSED_ADAM=1 pip install deepspeed==0.11.1

# RUN pip install notebook==6.4.8

ARG DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install openmim
RUN mim install mmengine mmcv

# Install MMAction2
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmaction2.git /mmaction2
WORKDIR /mmaction2
RUN mkdir -p /mmaction2/data
ENV FORCE_CUDA="1"
RUN git checkout 4d6c93474730cad2f25e51109adcf96824efc7a3
RUN pip install cython --no-cache-dir
RUN pip install --no-cache-dir -e .

# RUN git clone -b main https://github.com/open-mmlab/mmrazor.git /mmrazor
# WORKDIR /mmrazor

# RUN sed -i 's/--local_rank/--local-rank/g' /mmrazor/tools/train.py

# RUN pip install --no-cache-dir -v -e .

# RUN pip install --no-cache-dir pytorchvideo

WORKDIR /home/jovyan/

# RUN sed -i 's/num_classes=400,/num_classes=1001,/g' /mmaction2/configs/_base_/models/mvit_small.py
# RUN sed -i 's/num_classes=400,/num_classes=1001,/g' /mmaction2/configs/_base_/models/x3d.py
