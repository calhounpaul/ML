
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

model_loc = "/workdir/240618_200422_swift4_merged"

tokenizer = AutoTokenizer.from_pretrained(model_loc)
model = AutoModelForCausalLM.from_pretrained(model_loc)
model.to("cpu")

#get env var HF_TOKEN
hf_token = os.environ["HF_TOKEN"]

#push to huggingface pcalhoun/Mistral-7B-v0.3-JonathanSwift
model.push_to_hub("pcalhoun/Mistral-7B-v0.3-JonathanSwift", token=hf_token)
tokenizer.push_to_hub("pcalhoun/Mistral-7B-v0.3-JonathanSwift", token=hf_token)