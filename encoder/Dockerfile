# CUDA training environment for the place-description encoder (encoder/src).
# Build from this directory:  docker build -t lantern-encoder .
# Run with GPU:               docker run --gpus all -it -v "$PWD":/workspace lantern-encoder
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /workspace

COPY requirements.txt .

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install -r requirements.txt
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Optional: faster attention kernels for the DINOv2 backbone
RUN uv pip install xformers || true

CMD ["/bin/bash"]
