FROM transformers-quantization-latest-gpu

# Install the necessary packages
RUN apt update
RUN apt install -y git
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir opencv-python
RUN python3 -m pip install --no-cache-dir numpy
RUN python3 -m pip install --no-cache-dir pillow
RUN python3 -m pip install --no-cache-dir matplotlib