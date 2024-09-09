from PIL import Image
import requests, json, wget, zipfile, random, docker
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import sys
import pyvips
from urllib.parse import urlparse

THIS_FILE_PATH = os.path.abspath(__file__)
THIS_FILE_DIR = os.path.dirname(THIS_FILE_PATH)
MAX_IMAGES_COUNT = 9

def initialize_model_and_processor(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='flash_attention_2',
        load_in_8bit=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        num_crops=4
    )
    return model, processor

def generate_response(model, processor, messages, images):
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")
    
    generation_args = {
        "max_new_tokens": 1000,
        "temperature": 0.0,
        "do_sample": False,
    }
    
    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )
    
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return response

def is_url(path):
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def load_images(image_paths):
    images = []
    for path in image_paths:
        if is_url(path):
            response = requests.get(path, stream=True)
            response.raise_for_status()
            images.append(Image.open(response.raw))
        elif os.path.isfile(path):
            images.append(Image.open(path))
        else:
            print(f"Warning: Unable to load image from {path}. Skipping.")
    return images

def process_images_and_text(images, text):
    # Create a single placeholder string for all images
    image_placeholders = "".join(f"<|image_{i + 1}|>\n" for i in range(len(images)))
    
    # Combine image placeholders with the provided text
    full_content = image_placeholders + text
    
    # Create a single message with all content
    messages = [{"role": "user", "content": full_content.strip()}]

    print(json.dumps(messages, indent=2))
    return messages

def extract_ikea_pdf_pages(pdf_path, extracted_pages_path):
    #use pyvips to extract the pages from the pdf
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


if __name__ == "__main__":
    model_id = "microsoft/Phi-3.5-vision-instruct"
    path_to_cached_pdfs_and_images = "/root/.cache/cached_pdfs_and_images"
    downloaded_pdfs = [pdf for pdf in os.listdir(path_to_cached_pdfs_and_images) if pdf.lower().endswith(".pdf")]
    path_to_extracted_pages_dir = f"/root/.cache/extracted_pages/"
    if not os.path.exists(path_to_extracted_pages_dir):
        os.makedirs(path_to_extracted_pages_dir)
    paths_to_extracted_pages = []
    for pdf in downloaded_pdfs:
        pdf_name = pdf.split(".")[0]
        path_to_extracted_pages = os.path.join(path_to_extracted_pages_dir, pdf_name)
        if not os.path.exists(path_to_extracted_pages):
            os.makedirs(path_to_extracted_pages)
        path_to_pdf = os.path.join(path_to_cached_pdfs_and_images, pdf)
        extracted_pages = extract_ikea_pdf_pages(path_to_pdf, path_to_extracted_pages)
        paths_to_extracted_pages.append(extracted_pages)
    paths_and_replies = []
    for path in paths_to_extracted_pages:
        images = load_images([os.path.join(path, image) for image in os.listdir(path)])
        if len(images) == 0:
            print(f"Warning: No images found in {path}. Skipping.")
            continue
        if len(images) > MAX_IMAGES_COUNT:
            images_count = len(images)
            third_of_max=MAX_IMAGES_COUNT // 3
            exact_middle_index = images_count // 2
            first_images = images[:third_of_max]
            middle_images = images[exact_middle_index - third_of_max//2-1:exact_middle_index + third_of_max//2-1]
            last_images = images[-third_of_max:]
            images = first_images + middle_images + last_images
        prompt = """What is this instruction manual about?"""
        messages = process_images_and_text(images, prompt)
        model, processor = initialize_model_and_processor(model_id)
        response = generate_response(model, processor, messages, images)
        print("#" * 50)
        print("####",path)
        print("####",response)
        print("#" * 50)
        paths_and_replies.append({"manual_filename": path.split("/")[-1], "p35v_description": response})
    json.dump(paths_and_replies, open("furniture_assembly_manual_descriptions.json", "w"), indent=2)