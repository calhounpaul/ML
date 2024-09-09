THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
LIBRARIES_DIR_PATH=$PERMATHINGS_DIR_PATH/libs
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
GIT_AGGREGATES_PATH=$EPHEMERA_DIR_PATH/git_aggregates

if [ ! -d $EPHEMERA_DIR_PATH ]; then
    mkdir $EPHEMERA_DIR_PATH
fi
if [ ! -d $GIT_AGGREGATES_PATH ]; then
    mkdir $GIT_AGGREGATES_PATH
fi

if [ -z "$1" ]; then
    REPO_URL="https://github.com/calhounpaul/ML"
else
    REPO_URL=$1
fi

cd $LIBRARIES_DIR_PATH
python3 digest_git.py $REPO_URL --output_path $GIT_AGGREGATES_PATH

# --max_size_per_file 1000 \
#    --lower_bound_limit_per_folder 1 --max_folder_depth 3 --depth_augmented_limit_per_folder 5