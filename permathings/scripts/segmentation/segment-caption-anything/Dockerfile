FROM nvcr.io/nvidia/pytorch:23.07-py3

# Install the necessary packages
RUN apt update
RUN apt install -y git
RUN python3 -m pip install --no-cache-dir --upgrade pip

#make debian frontend noninteractive
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
RUN git clone https://github.com/xk-huang/segment-caption-anything
WORKDIR /app/segment-caption-anything
RUN . amlt_configs/setup.sh
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements-app.txt
RUN apt install -y git-lfs