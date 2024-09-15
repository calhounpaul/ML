THIS_DOCKER_CONTAINER_NAME="transformers-pixtral"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches

docker build -t $THIS_DOCKER_CONTAINER_NAME .

#if [ ! "$(docker ps -q -f name=$THIS_DOCKER_CONTAINER_NAME)" == "" ]; then
#    docker stop $THIS_DOCKER_CONTAINER_NAME
#fi

#if [ "$(docker ps -aq -f status=exited -f name=$THIS_DOCKER_CONTAINER_NAME)" ]; then
#    docker rm $THIS_DOCKER_CONTAINER_NAME
#fi

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
    -w /workdir \
    $THIS_DOCKER_CONTAINER_NAME
