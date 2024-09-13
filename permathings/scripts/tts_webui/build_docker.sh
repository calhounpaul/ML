#!/bin/bash

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBS_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
INPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/inputs
OUTPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/outputs
EBOOK_ANALYSES_DIR_PATH=$OUTPUTS_DIR_PATH/ebook_analyses
INPUT_EBOOKS_DIR_PATH=$INPUTS_DIR_PATH/ebooks
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
SECRETS_FILE_PATH=$EPHEMERA_DIR_PATH/secrets.json

cd $LIBS_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

# This script builds the vllm_server docker image
TEXT_GET_WEBUI_CACHE_FOLDER_PATH=$SHARED_CACHES_DIR_PATH/ttswui
if [ ! -d $TEXT_GET_WEBUI_CACHE_FOLDER_PATH ]; then
    mkdir -p $TEXT_GET_WEBUI_CACHE_FOLDER_PATH
fi

cd $TEXT_GET_WEBUI_CACHE_FOLDER_PATH

#remove text-generation-webui if it exists
#if [ -d "text-generation-webui-docker" ]; then
#    rm -rf text-generation-webui-docker
#fi

if [ ! -d "tts-generation-webui" ]; then
    git clone https://github.com/rsxdalv/tts-generation-webui
fi
cd tts-generation-webui

#mv docker-compose.yml docker-compose.yml.bak
#mv docker-compose.build.yml docker-compose.yml

#sed -i 's/        - VERSION_TAG="v1.5"/        - VERSION_TAG="nightly"/g' docker-compose.yml
#sed -i 's/      target: default/      target: default-nvidia/g' docker-compose.yml

#sed -i 's/                device_ids: \['0'\]/                device_ids: \['0', '1'\]/g' docker-compose.yml
#HF_TOKEN=$(jq -r '.HF_TOKEN' $SECRETS_FILE_PATH) docker compose up --build
#run compose build detach with secrets
HF_TOKEN=$HF_TOKEN docker build -t rsxdalv/tts-generation-webui .