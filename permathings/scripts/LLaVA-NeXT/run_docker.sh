THIS_DOCKER_CONTAINER_NAME="transformers-llava-next"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $(dirname $THIS_DIR_PATH))
TEMPLATE_DIR_PATH=$SCRIPTS_DIR_PATH/tformers_template
TEMPLATE_IMAGE_NAME="transformers-quantization-latest-gpu"
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches

#docker build -t $TEMPLATE_IMAGE_NAME $TEMPLATE_DIR_PATH
docker build -t $THIS_DOCKER_CONTAINER_NAME .

if [ ! -d $SHARED_CACHES_DIR_PATH ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH
fi

THIS_CACHE=$SHARED_CACHES_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME

if [ ! -d $THIS_CACHE ]; then
    mkdir -p $THIS_CACHE
fi

WORKDIR_PATH=$THIS_DIR_PATH/workspace

if [ ! -d $WORKDIR_PATH ]; then
    mkdir -p $WORKDIR_PATH
fi

docker kill $THIS_DOCKER_CONTAINER_NAME
docker rm $THIS_DOCKER_CONTAINER_NAME
sleep 2
docker kill $THIS_DOCKER_CONTAINER_NAME
docker rm $THIS_DOCKER_CONTAINER_NAME
sleep 2

docker run --gpus all -it --rm \
    --name $THIS_DOCKER_CONTAINER_NAME \
    -v $THIS_CACHE:/root/.cache \
    -v $WORKDIR_PATH:/workspace \
    -w /workspace \
    -p 7860:7860 \
    $THIS_DOCKER_CONTAINER_NAME
