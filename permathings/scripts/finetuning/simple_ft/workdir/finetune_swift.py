import json, os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training
from bitsandbytes.optim import Adam8bit
import datetime
import transformers
import bitsandbytes
from transformers.integrations import CodeCarbonCallback

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logging
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

RESUME_IF_CKPT = False

# Load dataset from JSON file
#YYMMDD_support_test
#TOKENIZERS_PARALLELISM=false
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#disable warnings
torch.autograd.set_detect_anomaly(True)
transformers.logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')


out_directory_name = "/workdir/"+datetime.datetime.now().strftime("%y%m%d") + "_swift2/"
training_data_path = "/workdir/dataset.json"

with open(training_data_path, "r") as f:
    train_data = json.load(f)

full_dataset = Dataset.from_dict({"text": train_data})

q4_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

q8_config = transformers.BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16,
)

# Load pre-trained model and tokenizer 
model_id = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id,max_length=4096)

config = AutoConfig.from_pretrained(model_id)

print("init empty weights")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

print("infer auto device map")

max_memory = {0:"5GiB", 1:"9GiB"} #,"cpu":"0GiB"}


print("max_memory", max_memory)

device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["MistralDecoderLayer"],
        dtype=torch.int8
    )


model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    #quantization_config=q8_config,
    quantization_config=q4_config,
    device_map=device_map,
    #device_map="balanced",
    token=os.environ.get("HF_TOKEN", None),
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
)

# Define new tokens
new_tokens = [
    "<|swift_context_start|>",
    "<|swift_context_end|>",
    "<|nonswift_context_start|>",
    "<|nonswift_context_end|>",
    "<|to_swiftify_start|>",
    "<|to_swiftify_end|>",
    "<|swift_start|>",
    "<|swift_end|>",
    ]
# Add new tokens to the tokenizer
num_added_tokens = tokenizer.add_tokens(new_tokens)
# Resize model's token embeddings to accommodate new tokens
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)

#add pad_token to tokenizer, </s>, for mistral
tokenizer.pad_token = tokenizer.eos_token

# Add LoRA adapters
lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj"], # + ["o_proj",] + ["gate_proj"], # + ["up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

#model = model_peft
model.print_trainable_parameters()

# Preprocess and tokenize data
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

full_dataset = full_dataset.map(preprocess, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

out_dir_path = os.path.join("/workdir/finetuned", out_directory_name)
if not os.path.exists(out_dir_path):
    os.makedirs(out_dir_path)

ckpt_dir=os.path.join(out_dir_path, "lora")
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

learning_rate = 1e-4

training_args = TrainingArguments(
    output_dir=ckpt_dir,
    #micro_batch_size=1,
    #per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    #gradient_checkpointing=True,
    num_train_epochs=1,
    learning_rate=learning_rate,
    #fp16=True,
    #bf16=True,
    logging_steps=1,
    save_strategy="steps",
    save_steps=40,
    save_total_limit=10,
    overwrite_output_dir=True,
    evaluation_strategy= "no",
    report_to="none",
)

optimizer = Adam8bit(model.parameters(), lr=learning_rate)

#clear cache before training
torch.cuda.empty_cache()

# Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_dataset,
    data_collator=data_collator,
)

trainer.remove_callback(CodeCarbonCallback)

resume_from_checkpoint = False
if os.path.exists(ckpt_dir) and RESUME_IF_CKPT:
    resume_from_checkpoint = True
# Fine-tune model
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

output_dir = os.path.join(out_dir_path, "output")

# Save fine-tuned model

try:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
except:
    print("Error saving model and tokenizer")
    pass