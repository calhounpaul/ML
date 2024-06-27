THIS_DIR_PATH=$(dirname $(realpath $0))
SCRIPTS_DIR_PATH=$(dirname $THIS_DIR_PATH)
PERMATHINGS_DIR_PATH=$(dirname $SCRIPTS_DIR_PATH)
ML_PROJECT_DIR_PATH=$(dirname $PERMATHINGS_DIR_PATH)
EPHEMERA_DIR_PATH=$ML_PROJECT_DIR_PATH/ephemera
INPUTS_DIR_PATH=$EPHEMERA_DIR_PATH/inputs
INPUT_EBOOKS_DIR_PATH=$INPUTS_DIR_PATH/ebooks
if [ ! -d $INPUT_EBOOKS_DIR_PATH ]; then
    mkdir -p $INPUT_EBOOKS_DIR_PATH
fi

for INPUT_EBOOK_PATH in $INPUT_EBOOKS_DIR_PATH/*.azw; do
    OUTPUT_EBOOK_PATH=$(echo $INPUT_EBOOK_PATH | sed 's/\.azw$/\.txt/')
    if [ -f $OUTPUT_EBOOK_PATH ]; then
        continue
    fi
    ebook-convert $INPUT_EBOOK_PATH $OUTPUT_EBOOK_PATH --enable-heuristics --replace-scene-breaks SCENE__BREAK__SCENE__BREAK
done


for INPUT_EBOOK_PATH in $INPUT_EBOOKS_DIR_PATH/*.epub; do
    OUTPUT_EBOOK_PATH=$(echo $INPUT_EBOOK_PATH | sed 's/\.epub$/\.txt/')
    if [ -f $OUTPUT_EBOOK_PATH ]; then
        continue
    fi
    ebook-convert $INPUT_EBOOK_PATH $OUTPUT_EBOOK_PATH --enable-heuristics --replace-scene-breaks SCENE__BREAK__SCENE__BREAK
done