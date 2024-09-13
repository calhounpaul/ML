from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "leafspark/Pixtral-12B-2409-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=True
)

print("Success")

input()

input("QUIT?")