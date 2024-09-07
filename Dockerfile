FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# set robot parameters
ENV ROBOT_TYPE=fr3
ENV LIBFRANKA_VERSION=0.10.0
ENV ROBOT_IP=172.16.0.1 
ENV NUC_IP=172.16.0.2
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,utility

# set directory structure
ARG NUC_ROBOT_CONFIG_DIR=/app/config/${ROBOT_TYPE}
ARG NUC_OCULUS_DIR=/app/droid/oculus_reader
ARG NUC_POLYMETIS_DIR=/app/droid/fairo/polymetis
ARG NUC_POLYMETIS_CONFIG_DIR=${NUC_POLYMETIS_DIR}/polymetis/conf

# copy project code to container
COPY . /app
WORKDIR /app

# base system installations
RUN apt-get update && \
    apt-get install -y software-properties-common build-essential sudo git curl wget python3-pip libspdlog-dev \
    libeigen3-dev lsb-release ffmpeg libsm6 libxext6 zstd && \
    apt-get upgrade -y

# Install Python 3.8.13.
RUN apt-get install -y \
        libbz2-dev \
        libffi-dev \
        liblzma-dev \
        libsqlite3-dev \
        libssl-dev  \
        wget \
        zlib1g-dev && \
    wget https://www.python.org/ftp/python/3.8.13/Python-3.8.13.tgz && \
    tar xvf Python-3.8.13.tgz && \
    cd Python-3.8.13 && \
    ./configure --enable-shared && \
    make -j8 && \
    make install



# Set LD_LIBRARY_PATH for Python 3.8.13.
ENV LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Install pip / wheel / gnureadline.
RUN python3.8 -m pip --no-cache-dir install --upgrade pip && \
    python3.8 -m pip --no-cache-dir install wheel && \
    apt-get install -y libncurses5-dev && \
    python3.8 -m pip --no-cache-dir install gnureadline




# install miniconda 
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH /root/miniconda3/bin:$PATH

# create conda environment
RUN conda create -n "robot" python=3.7
SHELL ["conda", "run", "-n", "robot", "/bin/bash", "-c"]

# install the zed sdk
ARG UBUNTU_RELEASE_YEAR=22
ARG ZED_SDK_MAJOR=4
ARG ZED_SDK_MINOR=1
ARG CUDA_MAJOR=12
ARG CUDA_MINOR=1

RUN echo "Europe/Paris" > /etc/localtime ; echo "CUDA Version ${CUDA_MAJOR}.${CUDA_MINOR}.0" > /usr/local/cuda/version.txt

# setup the ZED SDK
RUN apt-get update -y || true ; apt-get install --no-install-recommends lsb-release wget less udev sudo zstd build-essential cmake python3 python3-pip libpng-dev libgomp1 -y && \ 
    python3 -m pip install numpy opencv-python && \
    wget -q -O ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run https://download.stereolabs.com/zedsdk/${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}/cu${CUDA_MAJOR}${CUDA_MINOR%.*}/ubuntu${UBUNTU_RELEASE_YEAR} && \
    chmod +x ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run && \
    ./ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run -- silent skip_tools skip_cuda && \
    ln -sf /lib/x86_64-linux-gnu/libusb-1.0.so.0 /usr/lib/x86_64-linux-gnu/libusb-1.0.so && \
    rm ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run && \
    rm -rf /var/lib/apt/lists/*
RUN conda install -c conda-forge libstdcxx-ng requests # required for pyzed
RUN python /usr/local/zed/get_python_api.py 
RUN python -m pip install --ignore-installed /app/pyzed-4.1-cp37-cp37m-linux_x86_64.whl



# python environment setup
RUN pip3 install -e . && \
    pip3 install torch torchvision torchaudio && \
    pip3 install dm-robotics-moma==0.5.0 --no-deps && \
    pip3 install dm-robotics-transformations==0.5.0 --no-deps && \
    pip3 install dm-robotics-agentflow==0.5.0 --no-deps && \
    pip3 install dm-robotics-geometry==0.5.0 --no-deps && \
    pip3 install dm-robotics-manipulation==0.5.0 --no-deps && \
    pip3 install dm-robotics-controllers==0.5.0 --no-deps


# using miniconda instead of anaconda so overwrite sh scripts
RUN find /app/droid/franka -type f -name "launch_*.sh" -exec sed -i 's/anaconda/miniconda/g' {} \;
RUN find /app/scripts/server -type f -name "launch_server.sh" -exec sed -i 's/anaconda/miniconda/g' {} \;

# set absolute paths
RUN find /app/droid/franka -type f -name "launch_*.sh" -exec sed -i 's|~|/root|g' {} \;
RUN find /app/scripts/server -type f -name "launch_server.sh" -exec sed -i 's|~|/root|g' {} \;

# set polymetis config files
RUN cp ${NUC_ROBOT_CONFIG_DIR}/franka_hardware.yaml ${NUC_POLYMETIS_CONFIG_DIR}/robot_client/franka_hardware.yaml && \
    cp ${NUC_ROBOT_CONFIG_DIR}/franka_panda.yaml ${NUC_POLYMETIS_CONFIG_DIR}/robot_model/franka_panda.yaml

# start the server
RUN chmod +x /app/.docker/laptop/entrypoint.sh
ENTRYPOINT ["/app/.docker/laptop/entrypoint.sh"]