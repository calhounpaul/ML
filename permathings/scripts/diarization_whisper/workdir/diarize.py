import torch
import os
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# Configuration
WORKDIR_PATH = "/workdir"
REMOTE_INPUT_FILES_PATH = os.path.join(WORKDIR_PATH, "remote_input_files")
LOCAL_OUTPUT_FILES_PATH = os.path.join(WORKDIR_PATH, "diarization_output_files")
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE_NUM=1

def initialize_pipeline():
    """Initialize and configure the diarization pipeline with batching."""
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )
    # Set embedding batch size
    pipeline.embedding_batch_size = 192  # Adjust based on your GPU capacity
    
    # Use GPU if available
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda:"+str(DEVICE_NUM)))
    
    return pipeline

def load_all_mp3_files():
    """Recursively find all MP3 files in the input directory."""
    mp3_file_paths = []
    for root, _, files in os.walk(REMOTE_INPUT_FILES_PATH):
        for file in files:
            if file.endswith(".mp3"):
                mp3_file_paths.append(os.path.join(root, file))
    return sorted(mp3_file_paths)

def process_audio_file(pipeline, mp3_file_path, output_file_path):
    """Process a single audio file and save the diarization results."""
    print(f"Processing file: {mp3_file_path}")
    
    # Load the audio file
    waveform, sample_rate = torchaudio.load(mp3_file_path)
    
    # Process with progress monitoring
    with ProgressHook() as hook:
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            hook=hook
        )
    
    # Save the results
    with open(output_file_path, "w") as rttm:
        diarization.write_rttm(rttm)
    
    print(f"Diarization complete: {output_file_path}")

def main():
    # Initialize pipeline
    pipeline = initialize_pipeline()
    
    # Create output directory if it doesn't exist
    os.makedirs(LOCAL_OUTPUT_FILES_PATH, exist_ok=True)
    
    # Get all MP3 files
    all_mp3_files = load_all_mp3_files()
    
    # Process each file
    for mp3_file_path in all_mp3_files:
        # Create corresponding output directory structure
        relative_path = os.path.relpath(mp3_file_path, REMOTE_INPUT_FILES_PATH)
        output_dir = os.path.join(LOCAL_OUTPUT_FILES_PATH, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file path
        output_file_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(mp3_file_path))[0]}.rttm"
        )
        
        # Skip if file exists and is non-empty
        if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
            print(f"Skipping existing file: {output_file_path}")
            continue
        
        try:
            process_audio_file(pipeline, mp3_file_path, output_file_path)
        except Exception as e:
            print(f"Error processing {mp3_file_path}: {e}")

if __name__ == "__main__":
    main()
