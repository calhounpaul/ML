import requests
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig

import pyvips

def extract_pdf_pages(pdf_path, extracted_pages_path):
    #use pyvips to extract the pages from the pdfs
    pdf_filename = os.path.basename(pdf_path)
    noext_pdf_filename = ".".join(pdf_filename.split(".")[:-1])
    page_image_list = pyvips.Image.new_from_file(str(pdf_path)+"[dpi=300]")
    if not os.path.exists(extracted_pages_path):
        os.makedirs(extracted_pages_path)
    pdf_page_qty = page_image_list.get("n-pages")
    if pdf_page_qty == 0:
        print(f"Warning: No pages found in {pdf_path}. Skipping.")
        return extracted_pages_path
    if all([os.path.exists(os.path.join(extracted_pages_path, f"page-{i}.jpg")) for i in range(pdf_page_qty)]):
        print(f"Warning: All pages already exist in {pdf_path}. Skipping.")
        return extracted_pages_path
    for i in range(pdf_page_qty):
        print(f"Extracting page {i}...")
        page_image = pyvips.Image.new_from_file(pdf_path, page=i)
        page_image.write_to_file(os.path.join(extracted_pages_path, f"page-{i}.jpg"))
    return extracted_pages_path

if not os.path.exists("outputs/pdfs/puzzlesoldnew00hoff.pdf"):
    url = "https://dn790005.ca.archive.org/0/items/puzzlesoldnew00hoff/puzzlesoldnew00hoff.pdf"
    r = requests.get(url, allow_redirects=True)
    open("outputs/pdfs/puzzlesoldnew00hoff.pdf", "wb").write(r.content)

for root, dirs, files in os.walk("outputs/pdfs"):
    for file in files:
        if file.endswith(".pdf"):
            pdf_path = os.path.join(root, file)
            #extract to imgs/pdf_filename
            extracted_pages_path = os.path.join("outputs/imgs", os.path.basename(pdf_path).split(".")[0])
            if not os.path.exists(extracted_pages_path):
                os.makedirs(extracted_pages_path)
                extract_pdf_pages(pdf_path, extracted_pages_path)
            else:
                print(f"{extracted_pages_path} already exported as images. Skipping.")


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.bfloat16 if your GPU supports it
    bnb_4bit_quant_type="nf4",  # or "fp4" depending on your preference
    #bnb_4bit_use_double_quant=True,
)

#quantization_config = BitsAndBytesConfig(
#    load_in_8bit=True)

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
)
processor = AutoProcessor.from_pretrained(model_id)

image_paths = []
for root, dirs, files in os.walk("outputs/imgs"):
    for file in files:
        if file.endswith(".jpg"):
            image_paths.append(os.path.join(root, file))

for image_path in image_paths:
    image=Image.open(image_path)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Solve the puzzle(s) on the page. If there are no puzzles, say 'No puzzles found' and nothing else."}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=500)
    output_text = processor.decode(output[0])
    print(output_text)
    #save to file txt/pdf_filename/page-0.txt
    parent_path = os.path.dirname(image_path)
    #replace imgs/ with txt/
    parent_path = parent_path.replace("/imgs/", "/txt/")
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    txt_path = os.path.join(parent_path, f"{os.path.basename(image_path).split('.')[0]}.txt")
    print(f"Saving to {txt_path}")
    with open(txt_path, "w") as f:
        f.write(output_text)
