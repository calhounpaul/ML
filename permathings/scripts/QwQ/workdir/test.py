from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/QwQ-32B-Preview"
#model_name = "KirillR/QwQ-32B-Preview-AWQ"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="balanced",
    #device_map="auto",
    load_in_8bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Count the number of times the letter \"c\" appears in pneumonoultramicroscopicsilicovolcanoconiosis."
messages = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)

#save response to a txt file
with open('response.txt', 'w') as f:
    f.write(response)

