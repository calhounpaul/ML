# taken from https://huggingface.co/pyannote/speaker-diarization-3.1 - see for more details
# instantiate the pipeline
from pyannote.audio import Pipeline
import torch
import tempfile
import subprocess

pipeline = Pipeline.from_pretrained(
  "Revai/reverb-diarization-v2",
)
pipeline = pipeline.to(torch.device('cuda:0'))

import os

mp3s_folder = "/ephemera/podcast_files/"
diarizations_folder = "/ephemera/diarization/"

if not os.path.exists(diarizations_folder):
    os.makedirs(diarizations_folder)

def get_wav(mp3_path):
    tmp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(tmp_dir, "tmp.wav")
    subprocess.run(["ffmpeg", "-i", mp3_path, "-ac", "1", "-ar", "16000", wav_path])
    return wav_path

for mp3 in os.listdir(mp3s_folder):
    if mp3.endswith(".mp3"):
        diarization_path = os.path.join(diarizations_folder, mp3[:-4] + ".rttm")
        mp3_path = os.path.join(mp3s_folder, mp3)
        if not os.path.exists(diarization_path):
            print("Diarizing", mp3_path)
            wav = get_wav(mp3_path)
            print("tmp wav file", wav)
            diarization = pipeline(wav)
            with open(diarization_path + ".tmp", "w") as rttm:
                diarization.write_rttm(rttm)
            os.rename(diarization_path + ".tmp", diarization_path)
            print("Diarized", mp3_path)
        else:
            print("Already diarized", mp3_path)
