THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches
SDWUI_REPO_CACHE_PARENT=$SHARED_CACHES_DIR_PATH/stable_diffusion

SDWUI_DOCKER_REPO_URL="https://github.com/AbdBarho/stable-diffusion-webui-docker/"

#docker ps -a | grep "webui-docker" | awk '{print $1}' | xargs docker rm -f
#docker images | grep "webui-docker" | awk '{print $3}' | xargs docker rmi -f

if [ ! -d $EPHEMERA_DIR_PATH ]; then
	mkdir $EPHEMERA_DIR_PATH
fi
if [ ! -d $SHARED_CACHES_DIR_PATH ]; then
	mkdir $SHARED_CACHES_DIR_PATH
fi
if [ ! -d $SDWUI_REPO_CACHE_PARENT ]; then
	mkdir $SDWUI_REPO_CACHE_PARENT
fi
cd $SDWUI_REPO_CACHE_PARENT
if [ ! -d stable-diffusion-webui-docker ]; then
	git clone $SDWUI_DOCKER_REPO_URL
fi

cd stable-diffusion-webui-docker

git checkout 802d0bcd689e3a6fcdb56465c216caac01416816

docker compose --profile download up --build

sed -i 's/    image: sd-auto:78/#    image: sd-auto:78/g' docker-compose.yml

sed -i 's/git reset --hard v1.9.4 \&\& \\/git reset --hard v1.10.0 \&\& \\/g' services/AUTOMATIC1111/Dockerfile

if ! grep -q "typing_extensions" services/AUTOMATIC1111/requirements_versions.txt; then
	sed -i 's/pip install -r requirements_versions.txt/pip3 install --upgrade typing_extensions \&\& \\\n  pip install -r requirements_versions.txt/g' services/AUTOMATIC1111/Dockerfile
fi

cd $SDWUI_REPO_CACHE_PARENT/stable-diffusion-webui-docker

docker compose --profile auto up --build