import torch, os, sys, json, shutil
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
#model_id = "distil-whisper/distil-medium.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    load_in_8bit=True, device_map="balanced",
)
#model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    #device=device,
)


WORKDIR_PATH="/workdir"
REMOTE_INPUT_FILES_PATH=WORKDIR_PATH + "/remote_input_files"

#load all mp3 files in all directories and subdirectories
def load_all_mp3_files():
    mp3_file_paths = []
    for root, dirs, files in os.walk(REMOTE_INPUT_FILES_PATH):
        for file in files:
            if file.endswith(".mp3"):
                mp3_file_paths.append(os.path.join(root, file))
    return mp3_file_paths

all_mp3_files = load_all_mp3_files()
all_mp3_files.sort()

LOCAL_OUTPUT_FILES_PATH=WORKDIR_PATH + "/whisper_output_files"
if not os.path.exists(LOCAL_OUTPUT_FILES_PATH):
    os.makedirs(LOCAL_OUTPUT_FILES_PATH)

for mp3_file_path in all_mp3_files:
    print("Processing file: {}".format(mp3_file_path))
    #create the path of directories if not exists
    this_input_dir_path = os.path.dirname(mp3_file_path)
    this_output_dir_path = LOCAL_OUTPUT_FILES_PATH + "/" + this_input_dir_path.replace(REMOTE_INPUT_FILES_PATH, "")
    print("Checking for directory: {}".format(this_output_dir_path))
    if not os.path.exists(this_output_dir_path):
        os.makedirs(this_output_dir_path)
    #check if the file exists
    filename = os.path.basename(mp3_file_path)
    output_file_path = this_output_dir_path + "/" + filename + ".json"
    if os.path.exists(output_file_path):
        #check size > 10KB
        if os.path.getsize(output_file_path) > 10240:
            print("File exists and size > 10KB so skipping {}".format(output_file_path))
            continue
    print("Processing file: {}".format(mp3_file_path))
    result = pipe(mp3_file_path, batch_size=6, return_timestamps="word", chunk_length_s=60, stride_length_s=(5))
    with open(output_file_path, 'w') as f:
        json.dump(result, f, indent=4)
        print("File created: {}".format(output_file_path))