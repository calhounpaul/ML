THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $(dirname $THIS_DIR_PATH))
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
SHARED_CACHES_DIR_PATH=$EPHEMERA_DIR_PATH/shared_caches

if [ ! -d $SHARED_CACHES_DIR_PATH ]; then
    mkdir -p $SHARED_CACHES_DIR_PATH
fi
cd $SHARED_CACHES_DIR_PATH

git clone https://github.com/z-x-yang/Segment-and-Track-Anything.git

cd Segment-and-Track-Anything

#bash script/install.sh

mkdir ./ckpt

bash script/download_ckpt.sh

