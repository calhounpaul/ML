import os
from pathlib import Path
import logging
import json
import hashlib
import numpy as np
import soundfile as sf
from typing import Dict, List, Optional, NamedTuple, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import random
import librosa
from multiprocessing import Pool, cpu_count
import pickle
import shutil
from datasets import load_dataset, Audio, Features, Value, Sequence, Dataset

# Configure logging
def setup_logging(log_file: str = 'dataset_generator.log') -> logging.Logger:
    """Set up logging to both file and console with improved formatting."""
    logger = logging.getLogger('dataset_generator')
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

@dataclass
class DatasetConfig:
    """Configuration for dataset generation with improved defaults."""
    max_speakers: int = 4
    samples_per_count: int = 20000
    output_dir: str = "/workdir/diarized_speaker_dataset"
    audio_dir: str = "audio_files"
    combined_dir: str = "combined_audio"
    cache_dir: str = "processing_cache"
    min_duration: float = 2.0
    max_duration: float = 20.0
    min_overlap: float = 0.1
    max_overlap: float = 3.5
    sample_rate: int = 16000
    fade_duration: float = 0.05
    batch_size: int = 250

class AudioSegment(NamedTuple):
    """Representation of an audio segment with metadata."""
    start: float
    end: float
    speaker: str
    duration: float
    source_id: str
    audio: np.ndarray
    sr: int

class DiarizedAudioProcessor:
    def __init__(self, config: DatasetConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.segments_by_speaker = {}
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for output."""
        directories = [
            self.config.output_dir,
            os.path.join(self.config.output_dir, self.config.audio_dir),
            os.path.join(self.config.output_dir, self.config.combined_dir),
            os.path.join(self.config.output_dir, self.config.cache_dir)
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")

    def load_dataset_with_features(self, name: str, path: str, split: str, subset: Optional[str] = None) -> Optional[Dataset]:
        """Load dataset with proper feature specification."""
        try:
            features = Features({
                "audio": Audio(sampling_rate=self.config.sample_rate),
                "timestamps_start": Sequence(Value("float64")),
                "timestamps_end": Sequence(Value("float64")),
                "speakers": Sequence(Value("string"))
            })

            if subset:
                dataset = load_dataset(path, split, split=subset, features=features)
            else:
                dataset = load_dataset(path, split=split, features=features)

            self.logger.info(f"Successfully loaded dataset {name} with {len(dataset)} examples")
            return dataset
        except Exception as e:
            self.logger.error(f"Error loading dataset {name}: {e}")
            return None

    def process_audio_segment(self, audio_data: Dict, start: float, end: float,
                            speaker: str, source_id: str) -> Optional[AudioSegment]:
        """Process a single audio segment with improved error handling."""
        try:
            audio_array = audio_data['array']
            sample_rate = audio_data['sampling_rate']
            
            start_idx = int(start * sample_rate)
            end_idx = int(end * sample_rate)
            
            if start_idx >= len(audio_array) or end_idx > len(audio_array):
                return None
                
            segment_audio = audio_array[start_idx:end_idx].copy()
            duration = (end - start)
            
            # Resample if needed
            if sample_rate != self.config.sample_rate:
                segment_audio = librosa.resample(
                    y=segment_audio,
                    orig_sr=sample_rate,
                    target_sr=self.config.sample_rate
                )
            
            return AudioSegment(
                start=start,
                end=end,
                speaker=speaker,
                duration=duration,
                source_id=source_id,
                audio=segment_audio,
                sr=self.config.sample_rate
            )
        except Exception as e:
            self.logger.error(f"Error processing audio segment: {e}")
            return None

    def process_dataset(self, dataset: Dataset, name: str):
        """Process entire dataset with improved validation."""
        self.logger.info(f"Processing dataset: {name}")
        
        for idx, item in enumerate(dataset):
            try:
                # Validate required fields
                if not all(k in item for k in ['audio', 'timestamps_start', 'timestamps_end', 'speakers']):
                    self.logger.warning(f"Missing required fields in item {idx}")
                    continue
                    
                # Validate timestamps and speakers arrays
                if not (len(item['timestamps_start']) == len(item['timestamps_end']) == len(item['speakers'])):
                    self.logger.warning(f"Mismatched array lengths in item {idx}")
                    continue

                # Process segments
                for start, end, speaker in zip(
                    item['timestamps_start'],
                    item['timestamps_end'],
                    item['speakers']
                ):
                    duration = end - start
                    if self.config.min_duration <= duration <= self.config.max_duration:
                        segment = self.process_audio_segment(
                            item['audio'], start, end,
                            f"{name}_{idx}_{speaker}",
                            f"{name}_{idx}"
                        )
                        if segment:
                            speaker_id = f"{name}_{idx}_{speaker}"
                            if speaker_id not in self.segments_by_speaker:
                                self.segments_by_speaker[speaker_id] = []
                            self.segments_by_speaker[speaker_id].append(segment)
                            
            except Exception as e:
                self.logger.error(f"Error processing item {idx}: {e}")
                continue
                
        self.logger.info(f"Processed dataset {name}: found {len(self.segments_by_speaker)} unique speakers")

    def create_synthetic_example(self, num_speakers: int) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Create a synthetic example with multiple speakers."""
        if len(self.segments_by_speaker) < num_speakers:
            self.logger.warning(f"Insufficient speakers: {len(self.segments_by_speaker)} < {num_speakers}")
            return None
            
        try:
            selected_speakers = random.sample(list(self.segments_by_speaker.keys()), num_speakers)
            combined_audio = []
            speaker_order = []
            
            for speaker in selected_speakers:
                segments = self.segments_by_speaker[speaker]
                if not segments:
                    continue
                    
                segment = random.choice(segments)
                
                if combined_audio:
                    overlap = random.uniform(self.config.min_overlap, self.config.max_overlap)
                    overlap_samples = int(overlap * self.config.sample_rate)
                    if overlap_samples < len(combined_audio):
                        combined_audio = combined_audio[:-overlap_samples]
                        
                combined_audio.extend(segment.audio)
                speaker_order.append(speaker)
                
            if not combined_audio:
                return None
                
            return np.array(combined_audio), speaker_order
            
        except Exception as e:
            self.logger.error(f"Error creating synthetic example: {e}")
            return None

    def generate_dataset(self) -> List[Dict]:
        """Generate the complete dataset."""
        self.logger.info("Starting dataset generation")
        
        datasets_config = {
            'simsamu': ('diarizers-community/simsamu', 'train', None),
            'callhome_deu': ('talkbank/callhome', 'deu', 'data'),
            'callhome_eng': ('talkbank/callhome', 'eng', 'data')
        }
        
        # Load and process all datasets
        for name, (path, split, subset) in datasets_config.items():
            dataset = self.load_dataset_with_features(name, path, split, subset)
            if dataset:
                self.process_dataset(dataset, name)
                
        # Generate synthetic examples
        dataset_examples = []
        for speaker_count in range(1, self.config.max_speakers + 1):
            self.logger.info(f"Generating examples for {speaker_count} speakers")
            examples_generated = 0
            attempts = 0
            max_attempts = self.config.samples_per_count * 2
            
            while examples_generated < self.config.samples_per_count and attempts < max_attempts:
                result = self.create_synthetic_example(speaker_count)
                attempts += 1
                
                if result:
                    audio_array, speakers = result
                    
                    # Normalize audio
                    max_amplitude = np.max(np.abs(audio_array))
                    if max_amplitude > 0:
                        audio_array = audio_array / max_amplitude
                        
                    # Save audio file
                    audio_hash = hashlib.md5(audio_array.tobytes()).hexdigest()[:10]
                    filename = f"combined_{speaker_count}speakers_{audio_hash}.wav"
                    filepath = os.path.join(self.config.output_dir, self.config.combined_dir, filename)
                    
                    try:
                        sf.write(filepath, audio_array, self.config.sample_rate)
                        
                        example = {
                            "audio_path": str(Path(filepath).resolve()),
                            "num_speakers": speaker_count,
                            "speakers": speakers,
                            "duration": len(audio_array) / self.config.sample_rate
                        }
                        dataset_examples.append(example)
                        examples_generated += 1
                        
                        if examples_generated % 100 == 0:
                            self.logger.info(f"Generated {examples_generated}/{self.config.samples_per_count} "
                                           f"examples for {speaker_count} speakers")
                                           
                    except Exception as e:
                        self.logger.error(f"Error saving audio file: {e}")
                        continue
                        
            self.logger.info(f"Completed generating {examples_generated} examples for {speaker_count} speakers")
            
        # Save dataset metadata
        output_file = os.path.join(self.config.output_dir, "diarized_dataset.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(dataset_examples, f, indent=2)
            self.logger.info(f"Dataset metadata saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving dataset metadata: {e}")
            
        return dataset_examples

def main():
    """Main entry point."""
    logger = setup_logging()
    config = DatasetConfig()
    processor = DiarizedAudioProcessor(config, logger)
    
    try:
        dataset = processor.generate_dataset()
        logger.info(f"Successfully generated {len(dataset)} total examples")
    except Exception as e:
        logger.error(f"Fatal error during dataset generation: {e}")
        raise

if __name__ == "__main__":
    main()