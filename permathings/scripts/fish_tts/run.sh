#!/bin/bash

THIS_DOCKER_CONTAINER_NAME="fish-speech-tts"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches

cd $LIBRARIES_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

docker run -it \
    --name $THIS_DOCKER_CONTAINER_NAME \
    --gpus all \
    -p 7860:7860 \
    -e HF_TOKEN=$HF_TOKEN \
    fishaudio/fish-speech:latest-dev \
    zsh