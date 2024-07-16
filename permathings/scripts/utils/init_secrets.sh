THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera

if [ ! -d $EPHEMERA_DIR_PATH ]; then
    mkdir $EPHEMERA_DIR_PATH
fi

cd $LIBRARIES_DIR_PATH
python3 -c "from secretary import get_all_defaults; get_all_defaults()"
cd $THIS_DIR_PATH
