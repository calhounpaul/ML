FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git
RUN apt install -y curl
RUN apt install -y wget
RUN apt install -y python3.10-full
RUN apt install -y python3-pip python3-dev python3-venv
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade mistral-inference
RUN python3 -m pip install --no-cache-dir transformers huggingface_hub
RUN python3 -m pip install --no-cache-dir outlines
RUN python3 -m pip install --no-cache-dir pydantic
RUN python3 -m pip install --no-cache-dir protobuf
RUN python3 -m pip install --no-cache-dir accelerate
RUN python3 -m pip install --no-cache-dir bitsandbytes
RUN python3 -m pip install --no-cache-dir sentence-splitter
RUN python3 -m pip install --no-cache-dir Unidecode
WORKDIR /workspace