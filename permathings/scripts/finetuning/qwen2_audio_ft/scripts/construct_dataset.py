import os
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset
import json
import hashlib
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
from collections import defaultdict
import random
import librosa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THIS_DIR_PATH = Path(__file__).resolve().parent

@dataclass
class DatasetConfig:
    max_speakers: int = 3  # Maximum number of speakers to combine
    samples_per_count: int = 5000  # Number of examples to generate per speaker count
    output_dir: str = "speaker_count_dataset_2"
    audio_dir: str = "audio_files"
    combined_dir: str = "combined_audio"
    max_files_per_example: int = 3  # Maximum number of audio files to combine
    min_duration: float = 2.0  # Minimum duration in seconds
    max_duration: float = 14.0  # Maximum duration in seconds
    min_delay: float = 0.1  # Minimum delay between audio files
    max_delay: float = 10.0  # Maximum delay between audio files
    sample_rate: int = 16000  # Target sample rate for all audio
    min_upvotes: int = 0  # Minimum number of upvotes required
    max_downvotes: int = 0  # Maximum number of downvotes allowed

class AudioProcessor:
    @staticmethod
    def load_audio(file_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
        """Load audio file and resample if necessary."""
        audio, sr = sf.read(file_path)
        
        # Convert stereo to mono if necessary
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Ensure audio is float32 for consistent processing
        audio = audio.astype(np.float32)
        
        if sr != target_sr:
            audio = librosa.core.resample(
                audio,
                orig_sr=sr,
                target_sr=target_sr,
                res_type='scipy'
            )
            sr = target_sr
            
        return audio, sr

    @staticmethod
    def calculate_delays(audio_durations: List[float], target_duration: float) -> List[float]:
        """Calculate optimal delays for audio overlapping."""
        if not audio_durations:
            return []
            
        delays = [0.0]  # First file always starts at 0
        
        if len(audio_durations) == 1:
            return delays
            
        total_duration = sum(audio_durations)
        max_total_delay = min(
            target_duration - max(audio_durations),
            total_duration * 0.5
        )
        
        remaining_files = len(audio_durations) - 1
        avg_delay = max_total_delay / remaining_files
        
        for i in range(remaining_files):
            min_delay = avg_delay * 0.5
            max_delay = avg_delay * 1.5
            delay = random.uniform(min_delay, max_delay)
            delays.append(delay)
            
        return delays

    @staticmethod
    def combine_audio_with_delay(
        audio_files: List[str],
        delays: List[float],
        target_sr: int,
        target_duration: Optional[float] = None
    ) -> np.ndarray:
        """Combine multiple audio files with controlled overlapping."""
        # Load and preprocess all audio files
        audio_data = []
        for file_path in audio_files:
            audio, sr = AudioProcessor.load_audio(file_path, target_sr)
            
            # Apply fade in/out to prevent clicks
            fade_duration = min(0.05, len(audio) / target_sr / 4)
            fade_length = int(fade_duration * target_sr)
            
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            
            audio[:fade_length] *= fade_in
            audio[-fade_length:] *= fade_out
            
            audio_data.append(audio)

        # Calculate optimal delays if not provided
        if not delays:
            durations = [len(audio) / target_sr for audio in audio_data]
            delays = AudioProcessor.calculate_delays(
                durations,
                target_duration or max(durations) * 1.5
            )

        # Calculate total duration needed
        end_times = [
            start + len(audio)
            for audio, start in zip(
                audio_data,
                [int(d * target_sr) for d in delays]
            )
        ]
        total_samples = max(end_times)
        
        # Create output array
        combined = np.zeros(total_samples, dtype=np.float32)
        
        # Add each audio file at its delayed position
        for audio, delay in zip(audio_data, delays):
            start_pos = int(delay * target_sr)
            end_pos = start_pos + len(audio)
            combined[start_pos:end_pos] += audio
            
        # Apply peak normalization only if needed
        max_amplitude = np.max(np.abs(combined))
        if max_amplitude > 1.0:
            combined /= max_amplitude
            
        return combined

class RawPretrainedDatasetGenerator:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.speaker_to_files: Dict[str, List[Tuple[str, float, Dict]]] = defaultdict(list)
        self.file_hashes: Dict[str, str] = {}
        
    def setup_directories(self):
        """Create necessary directories for the dataset."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, self.config.audio_dir), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, self.config.combined_dir), exist_ok=True)

    def is_valid_sample(self, item: Dict) -> bool:
        """Check if a sample meets our quality criteria."""
        return (
            item['up_votes'] >= self.config.min_upvotes and
            item['down_votes'] <= self.config.max_downvotes and
            item['age'] and
            item['gender'] and
            item['client_id']
        )

    def load_common_voice(self):
        """Load and process Common Voice dataset with metadata filtering."""
        logger.info("Loading Common Voice dataset...")
        dataset = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="train", 
                             streaming=True, trust_remote_code=True,cache_dir=THIS_DIR_PATH / "cache")
        
        processed_speakers: Set[str] = set()
        current_samples = 0
        max_samples_needed = self.config.samples_per_count * self.config.max_speakers * 2
        
        for item in dataset:
            if current_samples >= max_samples_needed:
                break
                
            if not self.is_valid_sample(item):
                continue
                
            duration = len(item['audio']['array']) / item['audio']['sampling_rate']
            if not (self.config.min_duration <= duration <= self.config.max_duration):
                continue
                
            current_samples += 1
            self.process_audio_sample(item)
            processed_speakers.add(item['client_id'])
                
        logger.info(f"Processed {len(processed_speakers)} unique speakers")

    def process_audio_sample(self, item):
        """Process individual audio samples and store metadata."""
        audio_array = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        
        audio_hash = hashlib.md5(audio_array.tobytes()).hexdigest()
        filename = f"{audio_hash}.wav"
        filepath = os.path.join(self.config.output_dir, self.config.audio_dir, filename)
        
        if not os.path.exists(filepath):
            sf.write(filepath, audio_array, sampling_rate)
        
        duration = len(audio_array) / sampling_rate
        metadata = {
            'age': item['age'],
            'gender': item['gender'],
            'accent': item['accent']
        }
        self.speaker_to_files[item['client_id']].append((filename, duration, metadata))
        self.file_hashes[filename] = audio_hash

    def generate_raw_format(self, audio_files: List[str], speaker_metadata: List[Dict]) -> Dict:
        """Generate the raw pretrained model format."""
        full_path_string = str(THIS_DIR_PATH) + "/" + self.config.output_dir + "/" + self.config.combined_dir + "/"
        eos_token = "<|endoftext|>"
        return {
            "audio_file_paths": [full_path_string + f for f in audio_files],
            #"text": f"<|audio_bos|><|AUDIO|><|audio_eos|>DISTINCT VOICE COUNT: {len(descriptions)}\n" + "\n".join(descriptions) + eos_token,
            "text": f"<|audio_bos|><|AUDIO|><|audio_eos|>DISTINCT VOICE COUNT: {len(speaker_metadata)}" + eos_token,
        }

    def create_combined_audio(self, selected_files: List[str], target_speakers: int) -> str:
        """Create a combined audio file with random delays between segments."""
        delays = [random.uniform(self.config.min_delay, self.config.max_delay) 
                 for _ in range(len(selected_files))]
        
        combined_audio = AudioProcessor.combine_audio_with_delay(
            [os.path.join(self.config.output_dir, self.config.audio_dir, f) for f in selected_files],
            delays,
            self.config.sample_rate
        )
        
        combined_hash = hashlib.md5(combined_audio.tobytes()).hexdigest()
        combined_filename = f"combined_{target_speakers}speakers_{combined_hash}.wav"
        combined_path = os.path.join(self.config.output_dir, self.config.combined_dir, combined_filename)
        
        sf.write(combined_path, combined_audio, self.config.sample_rate)
        
        return combined_filename

    def generate_dataset(self):
        """Generate dataset examples with varying speaker counts."""
        logger.info("Generating dataset examples...")
        dataset = []
        
        for speaker_count in range(1, self.config.max_speakers + 1):
            examples = self.generate_examples_for_speaker_count(speaker_count)
            dataset.extend(examples)
            logger.info(f"Generated {len(examples)} examples for {speaker_count} speakers")
        
        output_file = os.path.join(self.config.output_dir, "raw_pretrained_dataset.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Dataset saved to {output_file}")
        return dataset

    def generate_examples_for_speaker_count(self, target_speaker_count: int) -> List[dict]:
        """Generate examples for a specific speaker count."""
        examples = []
        attempts = 0
        max_attempts = self.config.samples_per_count * 10
        
        while len(examples) < self.config.samples_per_count and attempts < max_attempts:
            attempts += 1
            
            use_combined = random.choice([True, False])
            
            if use_combined:
                example = self.create_combined_example(target_speaker_count)
            else:
                example = self.create_multiple_file_example(target_speaker_count)
                
            if example:
                examples.append(example)
        
        return examples

    def create_combined_example(self, target_speaker_count: int) -> Optional[Dict]:
        """Create an example with multiple speakers combined into a single audio file."""
        available_speakers = list(self.speaker_to_files.keys())
        if len(available_speakers) < target_speaker_count:
            return None
            
        selected_speakers = random.sample(available_speakers, target_speaker_count)
        selected_files = []
        selected_metadata = []
        
        for speaker in selected_speakers:
            speaker_files = self.speaker_to_files[speaker]
            if speaker_files:
                file_info = random.choice(speaker_files)
                selected_files.append(file_info[0])
                selected_metadata.append(file_info[2])
        
        if not selected_files:
            return None
            
        combined_filename = self.create_combined_audio(selected_files, target_speaker_count)
        
        return self.generate_raw_format([combined_filename], selected_metadata)

    def create_multiple_file_example(self, target_speaker_count: int) -> Optional[Dict]:
        """Create an example using multiple separate audio files."""
        available_speakers = list(self.speaker_to_files.keys())
        if len(available_speakers) < target_speaker_count:
            return None
            
        num_files = random.randint(1, min(self.config.max_files_per_example, target_speaker_count))
        selected_speakers = random.sample(available_speakers, target_speaker_count)
        
        speaker_groups = np.array_split(selected_speakers, num_files)
        combined_files = []
        all_metadata = []
        
        for group in speaker_groups:
            group_files = []
            group_metadata = []
            for speaker in group:
                speaker_files = self.speaker_to_files[speaker]
                if speaker_files:
                    file_info = random.choice(speaker_files)
                    group_files.append(file_info[0])
                    group_metadata.append(file_info[2])
            
            if group_files:
                combined_filename = self.create_combined_audio(group_files, len(group))
                combined_files.append(combined_filename)
                all_metadata.extend(group_metadata)
        
        if not combined_files:
            return None
            
        return self.generate_raw_format(combined_files, all_metadata)

def main():
    config = DatasetConfig()
    generator = RawPretrainedDatasetGenerator(config)
    
    generator.setup_directories()
    generator.load_common_voice()
    dataset = generator.generate_dataset()
    
    logger.info(f"Generated {len(dataset)} total examples")

if __name__ == "__main__":
    main()