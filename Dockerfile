FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Sistema base
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN pip install --no-cache-dir uv

WORKDIR /workspace

# Copia solo requirements para cache eficiente
COPY requirements.txt .

# Crear entorno con uv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install --upgrade pip
RUN uv pip install -r requirements.txt
RUN uv pip install xformers

# PyTorch con CUDA
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

CMD ["/bin/bash"]
