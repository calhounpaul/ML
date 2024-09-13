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
VLLM_REPO_CACHE_PATH=$SHARED_CACHES_DIR_PATH/vllm-pixtral

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

cd $THIS_DIR_PATH

#vllm serve mistralai/Pixtral-12B-2409 --tokenizer_mode mistral --limit_mm_per_prompt 'image=4' --max_num_batched_tokens 16384

docker run --rm -it \
    -v $HF_CACHE_FOLDER_PATH:/root/.cache/huggingface \
    --gpus='"device=0"' \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -v ./examples:/vllm-workspace/examples \
    -w /vllm-workspace/examples \
    --ipc=host \
    -m 40g \
    pixtral_vllm_server
# \
#    vllm serve mistralai/Pixtral-12B-2409 --tokenizer_mode mistral --limit_mm_per_prompt 'image=4'

#    --dtype auto \
#    --max-model-len 8000 \
#    --model mistralai/Pixtral-12B-2409