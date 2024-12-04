FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
        git tmux \
        libsndfile1-dev tesseract-ocr \
        espeak-ng python3 python3-pip \
        ffmpeg libvips poppler-utils

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir pytesseract

RUN python3 -m pip install -U \
        bitsandbytes \
        wget doclayout-yolo huggingface_hub \
        pyvips PyMuPDF Pillow \
        accelerate \
        auto-gptq chromadb pypdf pdf2image

RUN python3 -m pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

RUN python3 -m pip install -U numpy transformers pillow gekko pandas

RUN python3 -m pip install -U git+https://github.com/Dao-AILab/flash-attention.git

RUN python3 -m pip install -U autoawq