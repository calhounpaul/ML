THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $(dirname $THIS_DIR_PATH))
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches

#execute print(get_secret("HF_TOKEN")) from secretary in python
cd $LIBRARIES_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

HF_CACHE_FOLDER_PATH=$SHARED_CACHES_DIR_PATH/huggingface

#docker run --rm --gpus '"device=0,1"' \
docker run --rm --gpus '"device=1"' \
    -e HF_TOKEN=$HF_TOKEN \
    -v ./workdir:/workdir \
    -w /workdir \
    -v $HF_CACHE_FOLDER_PATH:/root/.cache/huggingface \
    --name merge \
    -it finetune 