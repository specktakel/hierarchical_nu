FROM python:3.8

WORKDIR /home/

# Basic tools
RUN apt-get update && apt-get install -y git \
    dpkg-dev make g++ gcc binutils libx11-dev \
    libxpm-dev libxft-dev libxext-dev gfortran \
    cmake

# Dealing with SSH access to GitHub 
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Install hierarchical_nu and dependencies
RUN --mount=type=ssh pip install git+ssh://git@github.com/cescalara/hierarchical_nu.git

# Install CmdStan
RUN install_cmdstan

