THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
SDWUI_REPO_CACHE_PARENT=$SHARED_CACHES_DIR_PATH/stable_diffusion

cd $LIBRARIES_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")

SDWUI_REPO_CACHE=$SDWUI_REPO_CACHE_PARENT/stable-diffusion-webui-docker
cd $SDWUI_REPO_CACHE
docker compose --profile auto down
docker compose --profile comfy down
cd $THIS_DIR_PATH