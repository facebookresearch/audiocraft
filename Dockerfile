FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt apt update &&\
    apt install -y \
    wget \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python-is-python3 \
    ffmpeg \
    libnvrtc11.2 \
    libtcmalloc-minimal4

RUN useradd -m -u 1000 ac
RUN --mount=type=cache,target=/root/.cache python -m pip install --upgrade pip wheel
ENV TORCH_COMMAND="pip install torch==2.0.1+cu118 torchaudio --extra-index-url https://download.pytorch.org/whl/cu118"
RUN --mount=type=cache,target=/root/.cache python -m $TORCH_COMMAND
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvrtc.so.11.2 /usr/lib/x86_64-linux-gnu/libnvrtc.so
USER 1000
RUN mkdir ~/.cache
RUN --mount=type=cache,target=/home/ac/.cache --mount=source=.,target=/home/ac/audiocraft python -m pip install -r /home/ac/audiocraft/requirements.txt
WORKDIR /home/ac/audiocraft