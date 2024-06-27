THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
HF_CACHE_FOLDER_PATH=$SHARED_CACHES_DIR_PATH/huggingface

cd $LIBRARIES_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")
cd $THIS_DIR_PATH

docker run -it -p 7862:7860 --platform=linux/amd64 --gpus '"device=1"' \
		-e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
        -v $HF_CACHE_FOLDER_PATH:/root/.cache/huggingface \
	registry.hf.space/stabilityai-stable-diffusion-3-medium:latest python app.py