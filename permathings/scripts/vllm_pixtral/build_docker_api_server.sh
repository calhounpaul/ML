#!/bin/bash

REPO_NAME="vllm-pixtral"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBS_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
INPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/inputs
OUTPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/outputs
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
VLLM_REPO_CACHE_PATH=$SHARED_CACHES_DIR_PATH/$REPO_NAME

cd $LIBS_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

if [ ! -d $SHARED_CACHES_DIR_PATH ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH
fi

#rm -rf $VLLM_REPO_CACHE_PATH

if [ ! -d $VLLM_REPO_CACHE_PATH ]; then
    cd $SHARED_CACHES_DIR_PATH
    git clone https://github.com/vllm-project/vllm $REPO_NAME
    cd $REPO_NAME
    sed -i 's/ENTRYPOINT \[/#ENTRYPOINT \[/g' Dockerfile
    #RUN apt-get update && apt-get install -y python3-distutils
    echo "RUN apt-get update && apt-get install -y python3-distutils" >> Dockerfile
    #echo "ENTRYPOINT [\"vllm\", \"serve\", \"mistralai/Pixtral-12B-2409\", \"--tokenizer_mode\", \"mistral\", \"--limit_mm_per_prompt\", \"image=4\", \"--max_num_batched_tokens\", \"16384\"]" >> Dockerfile
fi

cd $VLLM_REPO_CACHE_PATH

docker build --build-arg RUN_WHEEL_CHECK=false -t pixtral_vllm_server .