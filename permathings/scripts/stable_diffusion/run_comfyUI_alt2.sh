THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
COMFY_REPO=$SHARED_CACHES_DIR_PATH/stable_diffusion/comfy_alt2
if [ ! -d $COMFY_REPO ]; then
    mkdir -p $COMFY_REPO
fi
#https://github.com/calhounpaul/ComfyUI-Docker
if [ ! -d $COMFY_REPO/ComfyUI-Docker ]; then
    git clone https://github.com/calhounpaul/ComfyUI-Docker.git $COMFY_REPO/ComfyUI-Docker
fi
cd $COMFY_REPO/ComfyUI-Docker/cu124-megapak

docker compose up --build

cd $THIS_DIR_PATH