THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
PATH_TO_MODELS=$SHARED_CACHES_DIR_PATH/llama_cpp_server/models

MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf?download=true"
MODEL_PATH="$PATH_TO_MODELS/mistral-7b-v0.1.Q4_0.gguf"

if [ ! -d $SHARED_CACHES_DIR_PATH ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH
fi

if [ ! -d $PATH_TO_MODELS ]; then
    mkdir -p $PATH_TO_MODELS
fi

if [ ! -f $MODEL_PATH ]; then
    wget -O $MODEL_PATH $MODEL_URL
fi

#kill and/or remove/purge any containers named llama_cpp_server
docker kill llama_cpp_server
docker rm llama_cpp_server

docker run -d -p 8080:8080 -v $PATH_TO_MODELS:/models \
    --name llama_cpp_server \
    --gpus all ghcr.io/ggerganov/llama.cpp:server-cuda \
    -m /models/mistral-7b-v0.1.Q4_0.gguf \
     -c 2048 --host 0.0.0.0 --port 8080 --n-gpu-layers 99