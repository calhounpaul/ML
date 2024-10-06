THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
OLLAMA_CACHE_FOLDER_PATH=$SHARED_CACHES_DIR_PATH/ollama
OLLAMA_WEB_FOLDER_PATH=$SHARED_CACHES_DIR_PATH/ollama_web

docker kill open-webui
docker rm open-webui
sleep 3

docker run -d -p 3000:8080 -p 11434:11434 --gpus '"device=0"' -v $OLLAMA_CACHE_FOLDER_PATH:/root/.ollama \
    -v $OLLAMA_WEB_FOLDER_PATH:/app/backend/data --name open-webui --restart always \
    -e OLLAMA_HOST=0.0.0.0 \
    ghcr.io/open-webui/open-webui:ollama
