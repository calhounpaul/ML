import os
import logging
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import PeftModel
import soundfile as sf
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvalConfig:
    base_model_name: str = "Qwen/Qwen2-Audio-7B"
    adapter_path: str = "./qwen2audio-finetuned-241211_185205"
    test_data_path: str = "/workdir/scripts/speaker_count_dataset_2/raw_pretrained_dataset.json"
    num_eval_samples: int = 100
    target_sr: int = 16000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 10
    log_raw_io: bool = True  # Add this parameter

class ModelEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            config.base_model_name,
            trust_remote_code=True
        )
        
        logger.info("Loading base model...")
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            config.base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        logger.info("Loading fine-tuned adapter...")
        self.model = PeftModel.from_pretrained(
            self.model,
            config.adapter_path
        )
        self.model.eval()

    def load_and_process_audio(self, audio_paths: List[str]) -> np.ndarray:
        """Load and combine multiple audio files."""
        audios = []
        for path in audio_paths:
            try:
                audio, sr = librosa.load(path, sr=self.config.target_sr, mono=True)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                audios.append(audio)
            except Exception as e:
                logger.error(f"Error loading audio {path}: {str(e)}")
                continue
        
        if not audios:
            return np.zeros(self.config.target_sr // 4)
        
        return np.concatenate(audios) if len(audios) > 1 else audios[0]

    def evaluate_sample(self, item: Dict) -> Dict:
        """Evaluate a single sample and return results."""
        try:
            # Load and process audio
            combined_audio = self.load_and_process_audio(item["audio_file_paths"])
            
            # Create input prompt
            input_prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>DISTINCT VOICE COUNT:"
            
            if self.config.log_raw_io:
                logger.info("\n" + "="*80)
                logger.info("Input audio files:")
                for path in item["audio_file_paths"]:
                    logger.info(f"- {path}")
                logger.info("\nInput prompt:")
                logger.info(input_prompt)
                logger.info("\nGround truth:")
                logger.info(item["text"])
                logger.info("-"*80)
            
            # Prepare inputs
            inputs = self.processor(
                text=input_prompt,
                audios=combined_audio,
                return_tensors="pt",
                sampling_rate=self.config.target_sr
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )

            # Decode prediction
            predicted_text = self.processor.batch_decode(
                generated_ids[:, inputs["input_ids"].size(1):],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            if self.config.log_raw_io:
                logger.info("\nModel response:")
                logger.info(predicted_text)
                logger.info("="*80 + "\n")
            
            # Extract ground truth speaker count
            true_count = int(item["text"].split("DISTINCT VOICE COUNT: ")[1].replace("<|endoftext|>", "").split("\n")[0])
            # Try to extract predicted count
            try:
                pred_count = None
                if "DISTINCT VOICE COUNT:" in predicted_text:
                    pred_count = int(predicted_text.split("DISTINCT VOICE COUNT: ")[1].replace("<|endoftext|>", "").split("\n")[0])
                else:
                    # Try to find any number in the text
                    import re
                    numbers = re.findall(r'\d+', predicted_text)
                    if numbers:
                        pred_count = int(numbers[0])
            except:
                pred_count = None
            return {
                "ground_truth": item["text"],
                "prediction": predicted_text,
                "true_count": true_count,
                "predicted_count": pred_count,
                "count_match": true_count == pred_count if pred_count is not None else False,
                "singular_vs_many": (true_count == 1 and pred_count == 1) or (true_count > 1 and pred_count > 1),
            }
            
        except Exception as e:
            logger.error(f"Error evaluating sample: {str(e)}")
            return None

    def run_evaluation(self):
        """Run evaluation on test dataset."""
        # Load test data
        with open(self.config.test_data_path, 'r') as f:
            test_data = json.load(f)
        
        random.shuffle(test_data)

        # Select samples for evaluation
        eval_samples = test_data[:self.config.num_eval_samples]
        
        # Evaluate samples
        results = []
        correct_counts = 0
        total_evaluated = 0
        correct_singular_vs_many = 0
        
        for i, sample in enumerate(eval_samples):
            logger.info(f"Evaluating sample {i+1}/{len(eval_samples)}")
            result = self.evaluate_sample(sample)
            
            if result:
                results.append(result)
                if result["count_match"]:
                    correct_counts += 1
                total_evaluated += 1
                if result["singular_vs_many"]:
                    correct_singular_vs_many += 1
        
        # Calculate metrics
        accuracy = correct_counts / total_evaluated if total_evaluated > 0 else 0
        
        # Save results
        output_dir = Path(self.config.adapter_path)
        eval_results = {
            "accuracy": accuracy,
            "total_samples": total_evaluated,
            "correct_counts": correct_counts,
            "detailed_results": results
        }
        
        output_file = output_dir / "evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"\nEvaluation Results:")
        logger.info(f"Total samples evaluated: {total_evaluated}")
        logger.info(f"Correct count predictions: {correct_counts}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Correct singular vs. many predictions: {correct_singular_vs_many}")
        logger.info(f"Detailed results saved to: {output_file}")
        
        return eval_results

def main():
    config = EvalConfig()
    evaluator = ModelEvaluator(config)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()