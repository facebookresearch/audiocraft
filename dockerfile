# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# #Let's Use NVIDIA's CUDA base image with Python 3.9 and cuDNN 8
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Then Setup environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AUDIOCRAFT_CACHE_DIR=/app/cache

# # Set DEBIAN_FRONTEND to noninteractive to disable prompts
ENV DEBIAN_FRONTEND=noninteractive

# Build argument for timezone with default value 'Etc/UTC'
ARG TIMEZONE=Etc/UTC

# Furhet we Set the working directory
WORKDIR /app

#TO Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-venv \
    python3-pip \
    python3.9-dev \
    ffmpeg \
    build-essential \
    git \
    libsndfile1-dev \ 
    libffi-dev \                
    cython \                    
    tzdata \  
    && rm -rf /var/lib/apt/lists/*

# Preconfigure timezone
RUN ln -fs /usr/share/zoneinfo/${TIMEZONE} /etc/localtime && \
    echo "${TIMEZONE}" > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata


# Also, we Create a symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Then Upgrade pip
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python packages with --no-binary for pesq
RUN pip install --no-binary=pesq -r requirements.txt

# Copy the rest of the application code
COPY . .

# Install the package in editable mode
# COPY setup.py .
RUN pip install -e .

# (Optional) Install additional dependencies for optional features (e.g., watermarking)
# Uncomment the following line if you need to install extras
# RUN pip install -e '.[wm]'

# (Optional) Create a non-root user for enhanced security
# RUN useradd -m audiocraftuser && chown -R audiocraftuser:audiocraftuser /app
# USER audiocraftuser

# Expose necessary ports if this application serves a web interface or API though
# i can't find any
# EXPOSE 8000

# Define the default command to run the training script
CMD ["python", "train.py"]




