THIS_DOCKER_CONTAINER_NAME="transformers-ebook-processing"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBS_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
INPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/inputs
OUTPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/outputs
EBOOK_ANALYSES_DIR_PATH=$OUTPUTS_DIR_PATH/ebook_analyses
INPUT_EBOOKS_DIR_PATH=$INPUTS_DIR_PATH/ebooks
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
SECRETS_FILE_PATH=$EPHEMERA_DIR_PATH/secrets.json

HWID=$(hwid | cut -d ' ' -f 2)

if [ ! -d $OUTPUTS_DIR_PATH ]; then
    mkdir -p $OUTPUTS_DIR_PATH
fi
if [ ! -f $EBOOK_ANALYSES_DIR_PATH ]; then
    mkdir -p $EBOOK_ANALYSES_DIR_PATH
fi

cd $LIBS_DIR_PATH
python3 -c "from secretary import get_secret; n=get_secret('HF_TOKEN')"
cd $THIS_DIR_PATH

HF_TOKEN_ENCRYPTED=$(jq -r '.HF_TOKEN' $SECRETS_FILE_PATH)

HF_TOKEN=$(echo "$HF_TOKEN_ENCRYPTED" | openssl enc -d -aes-256-cbc -a -salt -pass pass:$HWID -pbkdf2)

docker build -t $THIS_DOCKER_CONTAINER_NAME .

if [ ! -d $SHARED_CACHES_DIR_PATH ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH
fi

if [ ! -d $SHARED_CACHES_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME
fi

WORKDIR_PATH=$THIS_DIR_PATH/workspace

if [ ! -d $WORKDIR_PATH ]; then
    mkdir -p $WORKDIR_PATH
fi

#echo "HF_TOKEN $HF_TOKEN"
docker run -it --gpus '"device=1"' --rm \
    --name $THIS_DOCKER_CONTAINER_NAME-2 \
    -v $SHARED_CACHES_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME:/ephemeral_cache \
    -v $WORKDIR_PATH:/workspace \
    -e HF_TOKEN=$HF_TOKEN \
    -v $INPUT_EBOOKS_DIR_PATH:/ebooks \
    -v $EBOOK_ANALYSES_DIR_PATH:/ebook_analyses \
    -w /workspace \
    $THIS_DOCKER_CONTAINER_NAME
