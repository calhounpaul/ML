THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
COMFY_CACHE=$SHARED_CACHES_DIR_PATH/stable_diffusion/comfy
if [ ! -d $COMFY_CACHE ]; then
    mkdir -p $COMFY_CACHE
fi

docker run -it --rm \
  --name comfyui \
  --gpus all \
  -p 8188:8188 \
  -v $COMFY_CACHE:/home/runner \
  -e CLI_ARGS="" \
  yanwk/comfyui-boot:cu121
