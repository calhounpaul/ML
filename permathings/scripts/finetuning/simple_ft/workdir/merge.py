
import subprocess as sp
import os
from threading import Thread , Timer
import sched, time

import torch, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

branch = "main"

from transformers import AutoTokenizer, TextGenerationPipeline
from transformers import GPTQConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import transformers

base_model_repo_and_name = "mistralai/Mistral-7B-v0.3"

lora_model_loc="/workdir/240618_200422_swift4/final_model"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_repo_and_name,
    #use_fast=True,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_repo_and_name,
    #load_in_8bit=True,
)

new_tokens = [
    "<|context|>",
    "<|plain|>",
    "<|swiftify|>",
]

num_added_tokens = tokenizer.add_tokens(new_tokens)

model.resize_token_embeddings(len(tokenizer))

#model = prepare_model_for_kbit_training(model)

#tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, lora_model_loc)

model = model.merge_and_unload()
model.save_pretrained("/workdir/240618_200422_swift4_merged")
tokenizer.save_pretrained("/workdir/240618_200422_swift4_merged")