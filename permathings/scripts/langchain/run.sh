THIS_DOCKER_CONTAINER_NAME="transformers-langchain-local"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches

cd $LIBRARIES_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

if [ ! -f ./Dockerfile ]; then
    wget https://raw.githubusercontent.com/huggingface/transformers/main/docker/transformers-quantization-latest-gpu/Dockerfile 
fi

docker build -t $THIS_DOCKER_CONTAINER_NAME .

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
    -v $WORKDIR_PATH:/root/workdir \
    -w /root/workdir \
    -e HF_TOKEN=$HF_TOKEN \
    $THIS_DOCKER_CONTAINER_NAME
