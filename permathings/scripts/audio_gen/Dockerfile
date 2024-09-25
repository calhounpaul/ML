FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg \
    bzip2 ca-certificates libglib2.0-0 libsm6 libxext6 libxrender1 mercurial openssh-client procps subversion wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up conda
ENV PATH=/opt/conda/bin:$PATH
ARG INSTALLER_URL_LINUX64="https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh"
ARG SHA256SUM_LINUX64="33442cd3813df33dcbb4a932b938ee95398be98344dff4c30f7e757cd2110e4f"

RUN wget "${INSTALLER_URL_LINUX64}" -O miniconda.sh -q && \
    echo "${SHA256SUM_LINUX64} miniconda.sh" > shasum && \
    sha256sum --check --status shasum && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh shasum && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Set up conda environment
RUN conda init bash && \
    conda create -n pdf2audio python=3.9 -y

# Set the shell to use bash
SHELL ["/bin/bash", "--login", "-c"]

# Activate conda environment and install packages
RUN echo "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate pdf2audio && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.4.0.* torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git pytesseract && \
    pip install --no-cache-dir flash_attn bitsandbytes accelerate imageio-ffmpeg diffusers opencv-python && \
    pip install --no-cache-dir git+https://github.com/kkroening/ffmpeg-python" > /tmp/install_packages.sh && \
    bash /tmp/install_packages.sh

# Clone and set up transformers
RUN echo "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate pdf2audio && \
    git clone https://github.com/huggingface/transformers && \
    cd transformers && \
    pip install --no-cache-dir -e .[dev-torch,testing,video] && \
    python setup.py develop" > /tmp/setup_transformers.sh && \
    bash /tmp/setup_transformers.sh

# Clone and set up PDF2Audio
RUN echo "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate pdf2audio && \
    git clone https://github.com/lamm-mit/PDF2Audio /PDF2Audio && \
    cd /PDF2Audio && \
    pip install --no-cache-dir -r requirements.txt" > /tmp/setup_pdf2audio.sh && \
    bash /tmp/setup_pdf2audio.sh

WORKDIR /PDF2Audio

#replace "    demo.launch()" with "    demo.launch(server_name="0.0.0.0")" in the file /PDF2Audio/app.py
RUN sed -i 's/demo.launch()/demo.launch(server_name="0.0.0.0")/g' /PDF2Audio/app.py

# Set the default command to activate the conda environment
CMD ["/bin/bash", "--login", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate pdf2audio && /bin/bash"]