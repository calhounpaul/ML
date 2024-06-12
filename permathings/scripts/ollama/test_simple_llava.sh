THIS_MODEL="llava:34b-v1.6"
DIR_SUFFIX="llava_34b_1.6"

THIS_MODEL="llava-phi3"
DIR_SUFFIX="phi3"

THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
EPHEMERAL_OUTPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/outputs
EPHEMERAL_INPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/inputs
if [ ! -d $EPHEMERAL_OUTPUTS_DIR_PATH ]; then
    mkdir -p $EPHEMERAL_OUTPUTS_DIR_PATH
fi
if [ ! -d $EPHEMERAL_INPUTS_DIR_PATH ]; then
    mkdir -p $EPHEMERAL_INPUTS_DIR_PATH
fi

TEST_DATA_CACHE_FOLDER_PATH=$EPHEMERAL_INPUTS_DIR_PATH/ollama_simple_test_data_$DIR_SUFFIX
if [ ! -d $TEST_DATA_CACHE_FOLDER_PATH ]; then
    mkdir -p $TEST_DATA_CACHE_FOLDER_PATH
fi

OUTPUT_DATA_CACHE_FOLDER_PATH=$EPHEMERAL_OUTPUTS_DIR_PATH/ollama_simple_test_output_$DIR_SUFFIX
if [ ! -d $OUTPUT_DATA_CACHE_FOLDER_PATH ]; then
    mkdir -p $OUTPUT_DATA_CACHE_FOLDER_PATH
fi

cd $TEST_DATA_CACHE_FOLDER_PATH
if [ -d "National_Geographic_Wallpapers" ]; then
    if [ $(ls -1q National_Geographic_Wallpapers | wc -l) -ne 506 ]; then
        rm -rf National_Geographic_Wallpapers
    fi
fi

if [ ! -d "National_Geographic_Wallpapers" ]; then
    mkdir -p National_Geographic_Wallpapers
    cd National_Geographic_Wallpapers
    wget -O National_Geographic_Wallpapers.zip "https://archive.org/compress/National_Geographic_Wallpapers/formats=JPEG"
    unzip National_Geographic_Wallpapers.zip
    rm National_Geographic_Wallpapers.zip
    find . -type f ! -name "*.jpg" -delete
fi

cd $TEST_DATA_CACHE_FOLDER_PATH

for file in National_Geographic_Wallpapers/*.jpg; do
    echo "Processing $file"
    json_string="{'model': '$THIS_MODEL', 'prompt': 'What is in this picture?', 'stream': False, 'images': [base64.b64encode(open('"$file"', 'rb').read()).decode('utf-8')]}).json()"
    echo $json_string
    command_string="import requests, base64, json; print(json.dumps(requests.post('http://localhost:11434/api/generate', json="$json_string", indent=2))"
    response=$(python3 -c "$command_string")
    echo $response | jq . > $OUTPUT_DATA_CACHE_FOLDER_PATH/$(basename $file).json
done