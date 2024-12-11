import os
import logging
from dataclasses import dataclass
from datetime import datetime
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import librosa
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
import numpy as np

DRY_RUN = False

# Global Constants
RANDOM_SEED = 42
EVAL_SPLIT_RATIO = 0.01
SAVE_STEPS = 1
SAVE_TOTAL_LIMIT = 5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioTrainingConfig:
    model_name: str = "Qwen/Qwen2-Audio-7B"
    output_dir: str = f"./qwen2audio-finetuned-{datetime.now().strftime('%y%m%d_%H%M%S')}"
    dataset_path: str = "/workdir/scripts/speaker_count_dataset_2/raw_pretrained_dataset.json"
    cache_dir: str = "./dataset_cache"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    gpu_memory_limit: str = "24GiB"
    cpu_memory_limit: str = "48GiB"
    target_mel_frames: int = 3000

class RawAudioDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], processor: Any, target_sr: int = 16000):
        self.processor = processor
        self.target_sr = target_sr
        self.target_mel_frames = 3000
        self.data = data
        logger.info(f"Initialized dataset with {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def load_audio(self, audio_path: str) -> np.ndarray:
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return audio
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {str(e)}")
            return np.zeros(self.target_sr // 4)

    def pad_or_trim_features(self, features: torch.Tensor, target_length: int = 3000) -> torch.Tensor:
        curr_length = features.shape[-1]
        if curr_length > target_length:
            return features[..., :target_length]
        elif curr_length < target_length:
            pad_length = target_length - curr_length
            return torch.nn.functional.pad(features, (0, pad_length))
        return features

    def process_audio_features(self, audio_features: torch.Tensor) -> torch.Tensor:
        if len(audio_features.shape) == 3:
            processed_features = []
            for i in range(audio_features.shape[0]):
                feat = audio_features[i]
                feat = self.pad_or_trim_features(feat, self.target_mel_frames)
                processed_features.append(feat)
            audio_features = torch.stack(processed_features)
            audio_features = audio_features.view(128, -1)
            audio_features = self.pad_or_trim_features(audio_features, self.target_mel_frames)
        else:
            audio_features = self.pad_or_trim_features(audio_features, self.target_mel_frames)
        return audio_features

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            item = self.data[idx]
            audios = [self.load_audio(path) for path in item["audio_file_paths"]]
            if len(audios) > 1:
                combined_audio = np.concatenate(audios)
                audios = [combined_audio]

            model_inputs = self.processor(
                text=item["text"],
                audios=audios,
                return_tensors="pt",
                padding=True,
                sampling_rate=self.target_sr
            )

            model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
            model_inputs["input_features"] = self.process_audio_features(model_inputs["input_features"])

            if len(model_inputs["feature_attention_mask"].shape) > 1:
                model_inputs["feature_attention_mask"] = model_inputs["feature_attention_mask"].view(-1)
            model_inputs["feature_attention_mask"] = model_inputs["feature_attention_mask"][:self.target_mel_frames]

            labels = model_inputs["input_ids"].clone()
            model_inputs["labels"] = labels

            return model_inputs

        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            raise

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_input_len = max(x["input_ids"].size(0) for x in batch)
    batch_size = len(batch)
    target_mel_frames = batch[0]["input_features"].shape[-1]

    collated = {
        "input_ids": torch.full((batch_size, max_input_len), 0, dtype=torch.long),
        "attention_mask": torch.zeros((batch_size, max_input_len), dtype=torch.long),
        "labels": torch.full((batch_size, max_input_len), -100, dtype=torch.long),
        "input_features": torch.zeros((batch_size, 128, target_mel_frames), dtype=torch.float),
        "feature_attention_mask": torch.zeros((batch_size, target_mel_frames), dtype=torch.long),
    }

    for i, item in enumerate(batch):
        input_len = item["input_ids"].size(0)
        collated["input_ids"][i, :input_len] = item["input_ids"]
        collated["attention_mask"][i, :input_len] = 1
        collated["labels"][i, :input_len] = item["labels"]
        collated["input_features"][i] = item["input_features"]
        collated["feature_attention_mask"][i] = item["feature_attention_mask"]

    return collated

def train():
    config = AudioTrainingConfig()

    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )

    with open(config.dataset_path, 'r') as f:
        data = json.load(f)

    train_data, eval_data = train_test_split(data, test_size=EVAL_SPLIT_RATIO, random_state=RANDOM_SEED)

    train_dataset = RawAudioDataset(train_data, processor)
    eval_dataset = RawAudioDataset(eval_data, processor)

    if DRY_RUN:
        logger.info("Running in DRY_RUN mode - will decode and display model inputs")
        debug_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
        test_batch = next(iter(debug_loader))

        logger.info("\nBatch shapes:")
        for k, v in test_batch.items():
            logger.info(f"{k}: {v.shape}")

        logger.info("\nDecoded sequences:")
        for i in range(len(test_batch["input_ids"])):
            decoded = processor.tokenizer.decode(test_batch["input_ids"][i])
            logger.info(f"\nSequence {i+1}:\n{decoded}")

        logger.info("\nDRY_RUN complete - exiting before model loading and training")
        return

    max_memory = {
        0: config.gpu_memory_limit,
        1: config.gpu_memory_limit,
        "cpu": config.cpu_memory_limit
    }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        config.model_name,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    model.tie_weights()
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=256,
        lora_alpha=512,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        logging_dir='./logs',
        logging_steps=1,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        ddp_find_unused_parameters=False,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    model.save_pretrained(config.output_dir)

if __name__ == "__main__":
    train()
