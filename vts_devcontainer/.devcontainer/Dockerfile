FROM ros:jazzy

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    net-tools \
    ros-jazzy-cv-bridge \
    ros-jazzy-message-filters \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Reuse Ubuntu 24.04's pre-created UID/GID 1000 account as the development
# user. This preserves bind-mount ownership on macOS and Linux hosts.
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN usermod --login $USERNAME --home /home/$USERNAME --move-home ubuntu \
    && groupmod --new-name $USERNAME ubuntu \
    && usermod --gid $USER_GID $USERNAME \
    && apt update && apt install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt autoremove -y && apt clean -y && rm -rf /var/lib/apt/lists/* \
    && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc \
    && echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc

# Install the exact uv-locked dependency graph into Jazzy's system Python.
# ROS Python extensions are compiled for that interpreter, so a separate
# managed Python or isolated virtual environment would be ABI-incompatible.
WORKDIR /opt/vts-python
COPY pyproject.toml uv.lock ./
RUN uv export --quiet --frozen --no-dev --no-emit-project --emit-index-url \
        --output-file requirements.lock \
    && uv pip sync --system --break-system-packages \
        --index-strategy unsafe-best-match --require-hashes requirements.lock \
    && rm requirements.lock

WORKDIR /

CMD ["bash"]
