import os
import logging
import torch
from pathlib import Path
from transformers import AutoProcessor, AutoTokenizer, Qwen2AudioForConditionalGeneration
from peft import PeftModel
import argparse

# Global Configuration
BASE_MODEL_NAME = "Qwen/Qwen2-Audio-7B"
ADAPTER_PATH = "/workdir/qwen2audio-finetuned-241216_030223"
OUTPUT_PATH = ADAPTER_PATH + "-merged"
REPO_ID = 
DEVICE_MAP = "auto"
TORCH_DTYPE = torch.float16
TRUST_REMOTE_CODE = True

if REPO_ID is None:
    raise ValueError("Please set REPO_ID to the desired HuggingFace Hub repository.")

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HuggingFace Token
hf_token = os.environ["HF_TOKEN"]

def merge_weights():
    """Merge LoRA weights with base model weights."""
    logger.info(f"Loading base model from {BASE_MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE
    )
    
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME,
        device_map=DEVICE_MAP,
        torch_dtype=TORCH_DTYPE,
        trust_remote_code=TRUST_REMOTE_CODE
    )
    
    logger.info(f"Loading adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    
    logger.info("Merging weights...")
    model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to {OUTPUT_PATH}...")
    model.save_pretrained(OUTPUT_PATH)
    processor.save_pretrained(OUTPUT_PATH)
    
    return model, processor

def push_to_hub(model, processor):
    """Push model and processor to HuggingFace Hub."""
    logger.info(f"Pushing model to {REPO_ID}...")
    model.push_to_hub(REPO_ID, token=hf_token)
    
    logger.info(f"Pushing processor to {REPO_ID}...")
    processor.push_to_hub(REPO_ID, token=hf_token)

def push_adapter():
    """Push LoRA adapter to HuggingFace Hub."""
    logger.info(f"Pushing adapter from {ADAPTER_PATH} to {REPO_ID}...")
    
    adapter = PeftModel.from_pretrained(
        Qwen2AudioForConditionalGeneration.from_pretrained(
            BASE_MODEL_NAME,
            device_map=DEVICE_MAP,
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=TRUST_REMOTE_CODE
        ),
        ADAPTER_PATH
    )
    
    adapter.push_to_hub(REPO_ID, token=hf_token)

def main():
    parser = argparse.ArgumentParser(description="Merge and push LoRA models to HuggingFace Hub")
    parser.add_argument("--merge", action="store_true",
                      help="Merge weights before pushing")
    
    args = parser.parse_args()
    
    if args.merge:
        # Merge weights and save
        model, processor = merge_weights()
        
        # Push merged model
        push_to_hub(model, processor)
    else:
        # Push adapter only
        push_adapter()

if __name__ == "__main__":
    main()