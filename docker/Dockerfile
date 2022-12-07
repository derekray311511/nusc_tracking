FROM osrf/ros:noetic-desktop-full

ENV NVIDIA_VISIBLE_DEVICES \ 
${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get -y update && apt-get install -y \
    iputils-ping \
    net-tools \
    wget \
    curl \
    build-essential \
    libgl1-mesa-dev \
    libglew-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libglm-dev \
    libfreetype6-dev \
    libglfw3-dev \
    libglfw3 \
    libglu1-mesa-dev \
    freeglut3-dev \
    pip \
    openssh-server \
    python3-tk
RUN export DISPLAY=":0.0"
ENTRYPOINT sudo service ssh restart && bash
ARG USERNAME=Student
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"
RUN echo "source /opt/ros/noetic/setup.sh" >> ~/.bashrc
RUN echo "export PS1=\"(container) \$PS1\"" >> ~/.bashrc

## For remote ssh usage
# RUN mkdir -p /home/Student/.ssh
# RUN sudo sed -ri 's/^PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
# RUN sudo sed -ri 's/^#PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
# RUN echo "root:root" | sudo chpasswd
# RUN echo "${USER}:passwd" | sudo chpasswd
# EXPOSE 22 

RUN pip install imageio

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8
RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install nuscenes-devkit

# For tracking
RUN pip install pandas==1.1.4
RUN pip install motmetrics==1.1.3
RUN pip install numpy==1.19.2
RUN pip install filterpy

# For visualization
RUN pip install pyyaml
RUN pip install rospkg

# Change to user mode
USER $USERNAME
RUN echo "source /opt/ros/noetic/setup.bash" > ~/.bashrc
WORKDIR /home/Student/Tracking