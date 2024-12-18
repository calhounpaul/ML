import os
import json
import random

CACHE_FOLDER = "/workdir/voxceleb_cache"
CONCATENATIONS_FOLDER = os.path.join(CACHE_FOLDER, "concatenations")
MAX_QTY = 300000

all_files = os.listdir(CONCATENATIONS_FOLDER)
same_files = [file for file in all_files if file.startswith("same_") and file.endswith(".wav") and os.path.getsize(os.path.join(CONCATENATIONS_FOLDER, file)) > 10]
different_files = [file for file in all_files if file.startswith("diff_") and file.endswith(".wav") and os.path.getsize(os.path.join(CONCATENATIONS_FOLDER, file)) > 10]

full_dataset = []
for file in same_files:
    speaker_id = file.split("_")[1]
    full_dataset.append({
        "text": f"<|audio_bos|><|AUDIO|><|audio_eos|>DISTINCT VOICE COUNT: 1\n<|endoftext|>",
        "audio_file_paths": [os.path.join(CONCATENATIONS_FOLDER, file)],
    })
for file in different_files:
    speaker_id_1, speaker_id_2 = file.split("_")[1], file.split("_")[2]
    full_dataset.append({
        "text": f"<|audio_bos|><|AUDIO|><|audio_eos|>DISTINCT VOICE COUNT: 2\n<|endoftext|>",
        "audio_file_paths": [os.path.join(CONCATENATIONS_FOLDER, file)],
    })

random.shuffle(full_dataset)

full_dataset=full_dataset[:MAX_QTY]
with open("/workdir/voxceleb_cache/dataset.json", "w") as f:
    json.dump(full_dataset, f, indent=4)

print(len(full_dataset), "items written to dataset.json")