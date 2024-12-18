import os
import json
import torchaudio
import multiprocessing
from datasets import load_dataset
from multiprocessing import Pool
from time import sleep
from pydub import AudioSegment, generators
import random
import tempfile

CACHE_FOLDER = "/workdir/voxceleb_cache"
JSON_FOLDER = os.path.join(CACHE_FOLDER, "json")
AUDIO_FOLDER = os.path.join(CACHE_FOLDER, "audio")
CONCATENATIONS_FOLDER = os.path.join(CACHE_FOLDER, "concatenations")

MAX_SAME_SPEAKER_CONCATS = 100000
MAX_DIFFERENT_SPEAKER_CONCATS = 100000

MAX_LENGTH_ON_CONCAT = 28 #seconds

def concat_wavs(wav_paths, output_path, max_length):
    combined = AudioSegment.empty()
    assert len(wav_paths) == 2
    length_wav_1 = AudioSegment.from_wav(wav_paths[0]).duration_seconds
    length_wav_2 = AudioSegment.from_wav(wav_paths[1]).duration_seconds
    skip_from_wav1_start = 0
    skip_from_wav2_end = 0
    if length_wav_1 + length_wav_2 > max_length:
        amount_to_cut = (length_wav_1 + length_wav_2) - max_length
        fraction_to_cut = amount_to_cut / (length_wav_1 + length_wav_2)
        skip_from_wav1_start = length_wav_1 * fraction_to_cut
        skip_from_wav2_end = length_wav_2 * fraction_to_cut
    combined += AudioSegment.from_wav(wav_paths[0])[int(skip_from_wav1_start * 1000):]
    combined += generators.Sine(440).to_audio_segment(duration=1000)
    combined += AudioSegment.from_wav(wav_paths[1])[:-int(skip_from_wav2_end * 1000)]
    combined.export(output_path, format="wav")

def main():
    speaker_counts = {}
    print("Concatenating all audio files")
    os.makedirs(CONCATENATIONS_FOLDER, exist_ok=True)
    speaker_ids = os.listdir(AUDIO_FOLDER)
    speaker_ids = [folder for folder in speaker_ids if os.path.isdir(os.path.join(AUDIO_FOLDER, folder))]
    paths_by_speaker = {spid: [] for spid in speaker_ids}
    for speaker_id in paths_by_speaker:
        speaker_audio_folder = os.path.join(AUDIO_FOLDER, speaker_id)
        audio_files = os.listdir(speaker_audio_folder)
        audio_files = [file for file in audio_files if file.endswith(".wav")]
        audio_files = [os.path.join(speaker_audio_folder, file) for file in audio_files]
        paths_by_speaker[speaker_id] = audio_files
    #get already concatenated files
    same_speaker_concats = len([file for file in os.listdir(CONCATENATIONS_FOLDER) if file.startswith("same_") and file.endswith(".wav") and os.path.getsize(os.path.join(CONCATENATIONS_FOLDER, file)) > 10])
    different_speaker_concats = len([file for file in os.listdir(CONCATENATIONS_FOLDER) if file.startswith("diff_") and file.endswith(".wav") and os.path.getsize(os.path.join(CONCATENATIONS_FOLDER, file)) > 10])
    while len(paths_by_speaker) and (same_speaker_concats < MAX_SAME_SPEAKER_CONCATS or different_speaker_concats < MAX_DIFFERENT_SPEAKER_CONCATS):
        #choose random max_length between 5 and MAX_LENGTH_ON_CONCAT
        max_length = random.randint(5, MAX_LENGTH_ON_CONCAT)
        random_speaker_1 = random.choice(list(paths_by_speaker.keys()))
        first_random_file = random.choice(paths_by_speaker[random_speaker_1])
        first_random_file_name_segment = first_random_file.split("/")[-1].split("_")[-1].split(".")[0]
        if same_speaker_concats < MAX_SAME_SPEAKER_CONCATS and len(paths_by_speaker[random_speaker_1]) > 1:
            second_random_file = random.choice(paths_by_speaker[random_speaker_1])
            while second_random_file == first_random_file:
                second_random_file = random.choice(paths_by_speaker[random_speaker_1])
            second_random_file_name_segment = second_random_file.split("/")[-1].split("_")[-1].split(".")[0]
            concat_filename = f"same_{first_random_file_name_segment}_{second_random_file_name_segment}.wav"
            if not os.path.exists(os.path.join(CONCATENATIONS_FOLDER, concat_filename)):
                concat_wavs([first_random_file, second_random_file], os.path.join(CONCATENATIONS_FOLDER, concat_filename), max_length)
                same_speaker_concats += 1
                paths_by_speaker[random_speaker_1].remove(first_random_file)
                paths_by_speaker[random_speaker_1].remove(second_random_file)
                if len(paths_by_speaker[random_speaker_1]) == 0:
                    del paths_by_speaker[random_speaker_1]
                random_speaker_1 = random.choice(list(paths_by_speaker.keys()))
                first_random_file = random.choice(paths_by_speaker[random_speaker_1])
                first_random_file_name_segment = first_random_file.split("/")[-1].split("_")[-1].split(".")[0]
        if different_speaker_concats < MAX_DIFFERENT_SPEAKER_CONCATS:
            random_speaker_2 = random.choice(list(paths_by_speaker.keys()))
            while random_speaker_2 == random_speaker_1:
                random_speaker_2 = random.choice(list(paths_by_speaker.keys()))
            second_random_file = random.choice(paths_by_speaker[random_speaker_2])
            second_random_file_name_segment = second_random_file.split("/")[-1].split("_")[-1].split(".")[0]
            concat_filename = f"diff_{first_random_file_name_segment}_{second_random_file_name_segment}.wav"
            concat_wavs([first_random_file, second_random_file], os.path.join(CONCATENATIONS_FOLDER, concat_filename), max_length)
            different_speaker_concats += 1
            paths_by_speaker[random_speaker_1].remove(first_random_file)
            paths_by_speaker[random_speaker_2].remove(second_random_file)
            if len(paths_by_speaker[random_speaker_1]) == 0:
                del paths_by_speaker[random_speaker_1]
            if len(paths_by_speaker[random_speaker_2]) == 0:
                del paths_by_speaker[random_speaker_2]
    print("Done concatenating all audio files")

if __name__ == "__main__":
    main()