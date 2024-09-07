#!/bin/bash

# Set non-interactive frontend
export DEBIAN_FRONTEND=noninteractive

# Download and install CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list
sudo apt-get update

# Install CUDA Toolkit 12.6
sudo apt-get -y install cuda-toolkit-12-6

# Install NVIDIA drivers (legacy kernel module flavor for wider compatibility)
sudo apt-get install -y cuda-drivers

# Clean up
sudo rm cuda-keyring_1.1-1_all.deb