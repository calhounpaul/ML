
THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
OLLAMA_CACHE_FOLDER_PATH=$SHARED_CACHES_DIR_PATH/ollama

docker kill ollama
docker rm ollama
docker kill /ollama
docker rm /ollama

docker run -d --rm --gpus all \
    -v $OLLAMA_CACHE_FOLDER_PATH:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker exec ollama ollama run llava:34b-v1.6