THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
SDWUI_REPO_CACHE_PARENT=$SHARED_CACHES_DIR_PATH/stable_diffusion
SDWUI_REPO_PATH=$SDWUI_REPO_CACHE_PARENT/stable-diffusion-webui-docker
EXTENSIONS_DIR_PATH=$SDWUI_REPO_PATH/data/config/auto/extensions
MODELS_DIR_PATH=$SDWUI_REPO_PATH/data/models
CONFIG_JSON_PATH=$SDWUI_REPO_PATH/data/config/auto/config.json

cd $LIBRARIES_DIR_PATH
HF_TOKEN=$(python3 -c "from secretary import get_secret; print(get_secret('HF_TOKEN'))")

SDWUI_REPO_CACHE=$SDWUI_REPO_CACHE_PARENT/stable-diffusion-webui-docker

if [ $(stat -c "%U" $EXTENSIONS_DIR_PATH) != $USER ]; then
    sudo chown -R $USER:$USER $EXTENSIONS_DIR_PATH -R
fi

cd $EXTENSIONS_DIR_PATH
if [ ! -d sd-webui-lobe-theme ]; then
    git clone https://github.com/lobehub/sd-webui-lobe-theme
fi
if [ ! -d stable-diffusion-webui-auto-tls-https ]; then
    git clone https://github.com/papuSpartan/stable-diffusion-webui-auto-tls-https
fi
if [ ! -d sd-webui-infinite-image-browsing ]; then
    git clone https://github.com/zanllp/sd-webui-infinite-image-browsing
fi
if [ ! -d sd-civitai-browser-plus ]; then
    git clone https://github.com/BlafKing/sd-civitai-browser-plus
fi

cd $SDWUI_REPO_CACHE_PARENT/stable-diffusion-webui-docker

if [ ! -f $CONFIG_JSON_PATH ]; then
    echo "{}" > $CONFIG_JSON_PATH
fi

if [ $(stat -c "%U" $CONFIG_JSON_PATH) != $USER ]; then
    sudo chown $USER:$USER $CONFIG_JSON_PATH
fi

API_KEY=$(jq -r '.custom_api_key // ""' $CONFIG_JSON_PATH)

if [ -z "$API_KEY" ] || [ ${#API_KEY} -le 5 ]; then
    read -p "Do you have a CivitAI API Key? (y/n): " HAS_KEY
    if [ "$HAS_KEY" = "y" ]; then
        read -p "Please enter your CivitAI API Key: " CIVITAI_API_KEY
        if [ ${#CIVITAI_API_KEY} -gt 5 ]; then
            jq --arg key "$CIVITAI_API_KEY" '.custom_api_key = $key' $CONFIG_JSON_PATH > tmp.$$.json
            sudo rm $CONFIG_JSON_PATH
            sudo mv tmp.$$.json $CONFIG_JSON_PATH
        else
            echo "The API key is too short. Skipping."
        fi
    else
        echo "Continuing without CivitAI API Key."
    fi
fi

if [ $(stat -c "%U" $CONFIG_JSON_PATH) != $USER ]; then
    sudo chown $USER:$USER $CONFIG_JSON_PATH
fi

docker compose --profile auto up
cd $THIS_DIR_PATH