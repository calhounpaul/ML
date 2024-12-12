import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ReformatterConfig:
    """Configuration for dataset reformatting."""
    input_cache_path: str = "/workdir/diarized_speaker_dataset/diarized_dataset.json"
    output_dir: str = "diarized_speaker_dataset"
    cache_dir: str = "processing_cache"
    output_filename: str = "formatted_dataset.json"

def setup_logging(log_file: str = 'dataset_reformatter.log') -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger('dataset_reformatter')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class DatasetReformatter:
    def __init__(self, config: ReformatterConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.setup_directories()
        
    def setup_directories(self):
        """Ensure all necessary directories exist."""
        os.makedirs(os.path.join(self.config.output_dir, self.config.cache_dir), exist_ok=True)
        
    def load_cache(self) -> Optional[List[Dict]]:
        """Load the existing dataset cache if available."""
        cache_path = self.config.input_cache_path
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.error(f"Error loading cache: {e}")
            return None
            
    def format_text(self, num_speakers: int) -> str:
        """Format the text field according to the specified template."""
        return f"<|audio_bos|><|AUDIO|><|audio_eos|>DISTINCT VOICE COUNT: {num_speakers}\n<|endoftext|>"
        
    def reformat_entry(self, entry: Dict) -> Dict:
        """Transform a single entry from the cache format to the desired output format."""
        return {
            "audio_file_paths": [entry["audio_path"]],
            "text": self.format_text(entry["num_speakers"])
        }
        
    def reformat_dataset(self) -> Optional[List[Dict]]:
        """Reformat the entire dataset."""
        cache_data = self.load_cache()
        if not cache_data:
            self.logger.error("No cache data available to reformat")
            return None
            
        self.logger.info(f"Reformatting {len(cache_data)} entries")
        reformatted_data = [self.reformat_entry(entry) for entry in cache_data]
        
        output_path = os.path.join(self.config.output_dir, self.config.output_filename)
        try:
            with open(output_path, 'w') as f:
                json.dump(reformatted_data, f, indent=2)
            self.logger.info(f"Reformatted dataset saved to {output_path}")
            return reformatted_data
        except Exception as e:
            self.logger.error(f"Error saving reformatted dataset: {e}")
            return None

def main():
    """Main entry point."""
    logger = setup_logging()
    config = ReformatterConfig()
    
    try:
        reformatter = DatasetReformatter(config, logger)
        dataset = reformatter.reformat_dataset()
        if dataset:
            logger.info(f"Successfully reformatted {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Fatal error during dataset reformatting: {e}")
        raise

if __name__ == "__main__":
    main()