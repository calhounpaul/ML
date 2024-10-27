THIS_DOCKER_CONTAINER_NAME="mochi-video"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
HF_CACHE=$SHARED_CACHES_DIR_PATH/$THIS_DOCKER_CONTAINER_NAME
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs

cd $LIBRARIES_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

if [ ! -d $SHARED_CACHES_DIR_PATH ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH
fi
if [ ! -d $HF_CACHE ]; then
    mkdir -p $HF_CACHE
fi

docker run -it -p 7860:7860 --platform=linux/amd64 --gpus '"device=1"' \
	-e HF_TOKEN=$HF_TOKEN \
    -v $HF_CACHE:/root/.cache \
	registry.hf.space/facebook-melodyflow:latest python demos/melodyflow_app.py --listen 0.0.0.0