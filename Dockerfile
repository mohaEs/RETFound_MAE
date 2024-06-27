FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3-dev \
    build-essential \
    libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies
RUN python3 -m pip install \
    torch==1.8.1 \
    torchvision==0.9.1 \
    torchaudio==0.8.1 \
    opencv-python==4.5.3.56 \
    pandas==0.25.3 \
    Pillow==8.3.1 \
    protobuf==3.17.3 \
    pycm==3.2 \
    pydicom==2.3.0 \
    scikit-image==0.17.2 \
    scikit-learn==0.24.2 \
    scipy==1.5.4 \
    tensorboard==2.6.0 \
    tensorboard-data-server==0.6.1 \
    tensorboard-plugin-wit==1.8.0 \
    timm==0.3.2 \
    tqdm==4.62.1

# Set the working directory
WORKDIR /workspace

COPY . /workspace

# Entry point (can be modified as needed)
CMD ["bash"]
