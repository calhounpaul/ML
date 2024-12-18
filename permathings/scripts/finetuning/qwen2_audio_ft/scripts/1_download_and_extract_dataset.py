import os
import json
import torchaudio
import multiprocessing
from datasets import load_dataset
from multiprocessing import Pool
from time import sleep

HF_TOKEN = os.getenv("HF_TOKEN")

# Define global variables
CACHE_FOLDER = "/workdir/voxceleb_cache"
JSON_FOLDER = os.path.join(CACHE_FOLDER, "json")
AUDIO_FOLDER = os.path.join(CACHE_FOLDER, "audio")
PROGRESS_FILE = os.path.join(CACHE_FOLDER, "progress.txt")
TMP_PROGRESS_FILE = os.path.join(CACHE_FOLDER, "progress_tmp.txt")
MAX_PARALLEL = 80

def initialize_cache():
    os.makedirs(JSON_FOLDER, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_progress(count):
    with open(TMP_PROGRESS_FILE, "w") as f:
        f.write(str(count))
    os.rename(TMP_PROGRESS_FILE, PROGRESS_FILE)

# Function to decode audio
def decode_audio(sample):
    audio, _ = torchaudio.load(sample["m4a"])
    return audio

# Function to process and cache dataset item
def process_and_cache_item(args):
    index, ds = args
    try:
        set_affinity()
        sample = ds[index]
        speaker_id = str(sample["cls"]).zfill(5)
        filename_base = f"voxceleb_{str(index).zfill(8)}"

        # Define file paths
        speaker_audio_folder = os.path.join(AUDIO_FOLDER, speaker_id)
        os.makedirs(speaker_audio_folder, exist_ok=True)
        wav_filename = os.path.join(speaker_audio_folder, filename_base + ".wav")

        speaker_json_folder = os.path.join(JSON_FOLDER, speaker_id)
        os.makedirs(speaker_json_folder, exist_ok=True)
        json_filename = os.path.join(speaker_json_folder, filename_base + ".json")

        # Skip processing if JSON file already exists
        if os.path.exists(json_filename) and os.path.getsize(json_filename) > 10:
            return  # Item already processed, skip it

        # Save audio
        if not os.path.exists(wav_filename) or os.path.getsize(wav_filename) < 100:
            audio = decode_audio(sample)
            torchaudio.save(wav_filename, audio, 16000)

        # Save JSON metadata
        json_data = {
            "cls": sample["cls"],
            "key": sample["__key__"],
            "url": sample["__url__"],
            "audio": wav_filename,
        }
        with open(json_filename, "w") as f:
            json.dump(json_data, f)
    except Exception as e:
        print(f"Error processing item {index}: {e}")

# Function to set CPU affinity for each worker
def set_affinity():
    pid = os.getpid()
    num_cores = multiprocessing.cpu_count()
    cpu_core = pid % num_cores  # Assign a core based on process ID
    os.sched_setaffinity(pid, {cpu_core})
    print(f"Worker process {pid} assigned to CPU core {cpu_core}")

if __name__ == "__main__":
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("gaunernst/voxceleb2-dev-wds", split="train")
    dataset_size = len(dataset)
    print("Dataset size:", dataset_size)

    # Initialize cache directories
    initialize_cache()

    # Load progress
    start_index = load_progress()
    print(f"Resuming from index: {start_index}")

    # Prepare arguments for multiprocessing
    args = [(i, dataset) for i in range(start_index, dataset_size)]

    # Use a multiprocessing pool to process the dataset
    print("Processing dataset...")
    with Pool(processes=MAX_PARALLEL) as pool:
        for i, _ in enumerate(pool.imap_unordered(process_and_cache_item, args), start=start_index):
            if (i + 1) % 100 == 0 or i + 1 == dataset_size:
                print(f"Processed {i + 1} items...")
                save_progress(i + 1)

    # Save final progress
    save_progress(dataset_size)
    print("Dataset processing complete.")
