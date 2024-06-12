#####needs fixing

SECRETS_FILE_PATH=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))))/secrets.json
HF_TOKEN=$(jq -r '.HF_TOKEN' $SECRETS_FILE_PATH)

HF_CACHE_FOLDER_PATH=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))))/cache/hf
TUNED_MODELS_FOLDER_PATH=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))))/cache/models/finetuned
DATASET_PATH=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))))/cache/exports/dataset.json
#workdir should be /workdir
docker run --gpus '"device=0"' \
    -e HF_TOKEN=$HF_TOKEN \
    -v ./workdir:/workdir \
    -w /workdir \
    -v $HF_CACHE_FOLDER_PATH:/root/.cache/huggingface \
    -v $TUNED_MODELS_FOLDER_PATH:/finetuned \
    -v $DATASET_PATH:/dataset.json \
    -it finetune