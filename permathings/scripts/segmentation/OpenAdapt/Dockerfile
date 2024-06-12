FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git
RUN apt install -y curl
RUN apt install -y wget
RUN apt install -y python3.10-full
RUN apt install -y python3-pip python3-dev python3-venv
RUN python3 -m pip install --no-cache-dir --upgrade pip
WORKDIR /app
RUN git clone https://github.com/LLaVA-VL/LLaVA-NeXT
WORKDIR /app/LLaVA-NeXT
RUN pip install -e ".[train]"