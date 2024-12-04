THIS_DOCKER_CONTAINER_NAME="qwq-inference"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
DATA_CACHES_PATH=$SHARED_CACHES_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME
HF_CACHE_PATH=$DATA_CACHES_PATH/hf_cache
IO_CACHE_PATH=$DATA_CACHES_PATH/io_cache
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs

docker build -t $THIS_DOCKER_CONTAINER_NAME .

if [ ! -d $HF_CACHE_PATH ]; then
    mkdir -p $HF_CACHE_PATH
fi
if [ ! -d $IO_CACHE_PATH ]; then
    mkdir -p $IO_CACHE_PATH
fi

cd $LIBRARIES_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

WORKDIR_PATH=$THIS_DIR_PATH/workdir

if [ ! -d $WORKDIR_PATH ]; then
    mkdir -p $WORKDIR_PATH
fi

#docker run -it --gpus device=1 --cpus="8.0" --rm \
docker run -it --gpus all --cpus="8.0" --rm \
    --name $THIS_DOCKER_CONTAINER_NAME \
    -v $HF_CACHE_PATH:/root/.cache \
    -v $IO_CACHE_PATH:/eph \
    -v $WORKDIR_PATH:/workdir \
    -w /workdir \
    -e HF_TOKEN=$HF_TOKEN \
    $THIS_DOCKER_CONTAINER_NAME
