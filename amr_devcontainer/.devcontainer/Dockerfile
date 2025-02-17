FROM osrf/ros:humble-desktop

# Install additional packages
RUN sudo apt update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt install -y lubuntu-desktop net-tools pip python3-tk ros-humble-xacro xrdp \
    # Clean up
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm /run/reboot-required*

# Create a non-root user
ARG USERNAME=ros
ARG PASSWORD=humble
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME -p $(openssl passwd $PASSWORD) \
    # [Optional] Add sudo support for the non-root user
    && apt update \
    && apt install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # Clean up
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && echo "source /usr/share/bash-completion/completions/git" >> /home/$USERNAME/.bashrc \
    && echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc

# Configure remote desktop
RUN adduser xrdp ssl-cert
RUN sed -i '3 a echo " \
export GNOME_SHELL_SESSION_MODE=Lubuntu\\n\
export XDG_SESSION_TYPE=x11\\n\
export XDG_CURRENT_DESKTOP=LXQt\\n\
export XDG_CONFIG_DIRS=/etc/xdg/xdg-Lubuntu:/etc/xdg\\n\
" > ~/.xsessionrc' /etc/xrdp/startwm.sh
EXPOSE 3389

# Install Python dependencies
COPY requirements.txt .
RUN pip install --disable-pip-version-check --no-cache-dir -U -r requirements.txt \
    && rm requirements.txt

# Start xrdp and a bash terminal
CMD service xrdp start ; bash