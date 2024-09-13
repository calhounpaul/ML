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

cd $LIBS_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

docker run --runtime nvidia --gpus all \
    -v $SHARED_CACHES_DIR_PATH/vllm-pixtral:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
     -it \
    --entrypoint vllm \
    vllm/vllm-openai:latest \
    serve mistralai/Pixtral-12B-2409 --tokenizer_mode mistral \
    --limit_mm_per_prompt 'image=4' --enable-chunked-prefill False \
    --kv-cache-dtype fp8 --max-model-len 5500 --cpu-offload-gb 5
