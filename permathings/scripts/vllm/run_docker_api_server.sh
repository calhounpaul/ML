#!/bin/bash

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBS_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
INPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/inputs
OUTPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/outputs
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
HF_CACHE_FOLDER_PATH=$SHARED_CACHES_DIR_PATH/huggingface
VLLM_REPO_CACHE_PATH=$SHARED_CACHES_DIR_PATH/vllm

if [ ! -f $EPHEMERA_DIR_PATH/secrets.json ]; then
    echo "Secrets file not found. Running init_secrets.sh..."
    $PERMATHINGS_DIR_PATH/scripts/utils/init_secrets.sh
fi

if [ ! -d $SHARED_CACHES_DIR_PATH ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH
fi
if [ ! -d $HF_CACHE_FOLDER_PATH ]; then
    mkdir -p $HF_CACHE_FOLDER_PATH
fi

cd $LIBS_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")

cd $VLLM_REPO_CACHE_PATH

docker run --rm -it \
    -v $HF_CACHE_FOLDER_PATH:/root/.cache/huggingface \
    --gpus='"device=0"' \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    --ipc=host \
    -m 40g \
    outlines_vllm_server \
    --guided-decoding-backend outlines \
    --dtype auto \
    --max-model-len 8000 \
    --model mistralai/Mistral-7B-Instruct-v0.3