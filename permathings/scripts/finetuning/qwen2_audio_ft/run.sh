THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $(dirname $THIS_DIR_PATH))
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
FINETUNING_WORKDIR=$EPHEMERA_DIR_PATH/finetuning_workdir
SCRIPTS_DIR=$THIS_DIR_PATH/scripts

if [ ! -d $FINETUNING_WORKDIR ]; then
    mkdir -p $FINETUNING_WORKDIR
fi

#execute print(get_secret("HF_TOKEN")) from secretary in python
cd $LIBRARIES_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

HF_CACHE_FOLDER_PATH=$SHARED_CACHES_DIR_PATH/huggingface

#docker run --rm --gpus '"device=0,1"' \
#docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
#docker run --rm --runtime=nvidia --gpus all --cpus 8 \
docker run --rm --runtime=nvidia --gpus '"device=0,1"' --cpus 8 \
    -e HF_TOKEN=$HF_TOKEN \
    -v $FINETUNING_WORKDIR:/workdir \
    -v $SCRIPTS_DIR:/workdir/scripts \
    -w /workdir \
    -v $HF_CACHE_FOLDER_PATH:/root/.cache/huggingface \
    -it finetune_audio
