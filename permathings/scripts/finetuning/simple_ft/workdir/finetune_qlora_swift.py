#https://medium.com/@ogbanugot/notes-on-fine-tuning-llama-2-using-qlora-a-detailed-breakdown-370be42ccca1

import os, json, datetime
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from bitsandbytes.optim import Adam8bit
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logging
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

DEVICE_MAP="auto"
MODEL_ID="mistralai/Mistral-7B-v0.3"
USE_4BIT=True
USE_8BIT=False
BNB_4BIT_QUANT_TYPE="nf4"
COMPUTE_DTYPE=torch.bfloat16
USE_NESTED_QUANT=True

FP16=False
BF16=True
MAX_GRAD_NORM=0.3
MAX_STEPS=-1
WARMUP_RATIO=0.05
GROUP_BY_LENGTH=True
PACKING=False

NEW_TOKENS = [
    "<|context|>",
    "<|plain|>",
    "<|swiftify|>",
]

OUTPUT_DIR = "/workdir/"+datetime.datetime.now().strftime("%y%m%d_%H%M%S")+"_swift4"
TRAIN_DATA_PATH = "/workdir/dataset.json"

NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
SAVE_STEPS = 20
LOGGING_STEPS = 1
LEARNING_RATE = 6e-5
WEIGHT_DECAY = 0.001
LR_SCHEDULER_TYPE = "linear"

MAX_SEQ_LENGTH=1024

LORA_ALPHA=512
LORA_DROPOUT=0.07
LORA_R=256
TARGET_MODULES=["q_proj", "k_proj", "v_proj"] + ["o_proj",] + ["gate_proj"] # + ["up_proj","down_proj"]

with open(TRAIN_DATA_PATH, "r") as f:
    train_data = json.load(f)

full_dataset = Dataset.from_dict({"text": train_data})

model_id = MODEL_ID

"""
config = AutoConfig.from_pretrained(model_id)

print("init empty weights")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

print("infer auto device map")

max_memory = {0:"4GiB", 1:"4.5GiB","cpu":"0GiB"}


print("max_memory", max_memory)

device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["MistralDecoderLayer"],
        dtype=torch.int8
    )
"""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    load_in_8bit=USE_8BIT,
    bnb_8bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=USE_NESTED_QUANT,
)

training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim="adamw_bnb_8bit",
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    fp16=FP16,
    bf16=BF16,
    max_grad_norm=MAX_GRAD_NORM,
    max_steps=MAX_STEPS,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=GROUP_BY_LENGTH,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="none"
)

peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

config = AutoConfig.from_pretrained(model_id)

num_added_tokens = tokenizer.add_tokens(NEW_TOKENS)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE_MAP,
    #device_map=device_map,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer.add_special_tokens({'pad_token': '</s>'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)

#model = get_peft_model(model, peft_config)

trainer = SFTTrainer(
    model=model,
    train_dataset=full_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=PACKING,
)

trainer.train()

trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))