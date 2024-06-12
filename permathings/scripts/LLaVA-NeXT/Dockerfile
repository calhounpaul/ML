FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git
RUN apt install -y curl
RUN apt install -y wget
RUN apt install -y python3.10-full
RUN apt install -y python3-pip python3-dev python3-venv
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN apt install -y tesseract-ocr libtesseract-dev

WORKDIR /app
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
RUN git clone https://github.com/OpenAdaptAI/OpenAdapt.git
WORKDIR /app/OpenAdapt
ENV PYTHONPATH "${PYTHONPATH}:/app/OpenAdapt"
RUN python3 -m pip install --no-cache-dir poetry
RUN poetry install
RUN poetry run install-dashboard
RUN python3 -m pip install --no-cache-dir .[all]
WORKDIR /app/OpenAdapt/openadapt
RUN alembic upgrade head
RUN pytest