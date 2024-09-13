#!/bin/bash

REPO_NAME="vllm-outlines"
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

if [ ! -d $VLLM_REPO_CACHE_PATH ]; then
    cd $SHARED_CACHES_DIR_PATH
    git clone https://github.com/vllm-project/vllm $REPO_NAME
    cd $REPO_NAME
    echo "typing-extensions" >> requirements-common.txt
    git checkout 919770957f26d71a5a6eda7a1a7443dfeb5ba0ee
    sed -i 's/ENTRYPOINT \[/#ENTRYPOINT \[/g' Dockerfile
    echo "ENTRYPOINT [\"python3\", \"-m\", \"outlines.serve.serve\"]" >> Dockerfile
fi

cd $VLLM_REPO_CACHE_PATH

docker build --build-arg RUN_WHEEL_CHECK=false -t outlines_vllm_server .