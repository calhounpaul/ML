THIS_DOCKER_CONTAINER_NAME="mochi-video-2"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
COMFY_CACHE=$SHARED_CACHES_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME

docker build -t $THIS_DOCKER_CONTAINER_NAME .

if [ ! -d $SHARED_CACHES_DIR_PATH ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH
fi
if [ ! -d $COMFY_CACHE ]; then
    mkdir -p $COMFY_CACHE
fi

WORKDIR_PATH=$THIS_DIR_PATH/workspace

if [ ! -d $WORKDIR_PATH ]; then
    mkdir -p $WORKDIR_PATH
fi

docker run -it --gpus '"device=1"' --rm \
    --name $THIS_DOCKER_CONTAINER_NAME \
    -v $COMFY_CACHE:/root \
    -e CLI_ARGS="" \
    -p 8189:8188 \
    $THIS_DOCKER_CONTAINER_NAME
