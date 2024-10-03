THIS_DOCKER_CONTAINER_NAME="transformers-llama32-vision"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
OUTPUTS_CACHE_PATH=$EPHEMERA_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs

docker build -t $THIS_DOCKER_CONTAINER_NAME .

if ! [ -d $OUTPUTS_CACHE_PATH ]; then
    mkdir -p $OUTPUTS_CACHE_PATH
fi
#make pdfs, imgs, and txt dirs
if ! [ -d $OUTPUTS_CACHE_PATH/pdfs ]; then
    mkdir -p $OUTPUTS_CACHE_PATH/pdfs
fi
if ! [ -d $OUTPUTS_CACHE_PATH/imgs ]; then
    mkdir -p $OUTPUTS_CACHE_PATH/imgs
fi
if ! [ -d $OUTPUTS_CACHE_PATH/txt ]; then
    mkdir -p $OUTPUTS_CACHE_PATH/txt
fi

cd $LIBRARIES_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

if [ ! -d $SHARED_CACHES_DIR_PATH ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH
fi

if [ ! -d $SHARED_CACHES_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME
fi

WORKDIR_PATH=$THIS_DIR_PATH/workdir

if [ ! -d $WORKDIR_PATH ]; then
    mkdir -p $WORKDIR_PATH
fi

docker run -it --gpus all --rm \
    --name $THIS_DOCKER_CONTAINER_NAME \
    -v $SHARED_CACHES_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME:/root/.cache \
    -v $WORKDIR_PATH:/workdir \
    -v $OUTPUTS_CACHE_PATH:/workdir/outputs \
    -w /workdir \
    -e HF_TOKEN=$HF_TOKEN \
    $THIS_DOCKER_CONTAINER_NAME
