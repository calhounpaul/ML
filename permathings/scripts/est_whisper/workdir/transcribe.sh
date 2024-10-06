mkdir -p /ephemera/transcribed_episodes

#repeat ten times
cd /ephemera/transcribed_episodes

for i in {1..10}; do
    for f in /ephemera/podcast_files/*.mp3; do
        if [ ! -f /ephemera/transcribed_episodes/$(basename $f .mp3).txt ]; then
            whisper-ctranslate2 --model_directory /ephemera/whisper-large-et.ct2 --task transcribe --language et --beam_size 5 $f
        fi
    done
done

python3 diarize.py
