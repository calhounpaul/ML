import os, sys, re

#cuda visible devices= 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from pydantic import BaseModel
import json
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logging
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
torch.cuda.empty_cache()
import outlines


MODEL_DIR_PATH="/ephemeral_cache/mistral-inst-v03"


function_json_string = '''{
    "type": "function",
    "function": {
        "name": "store_text_outline",
        "description": "Store the outline of a text in a hierarchical manner",
        "parameters": {
            "type": "object",
            "properties": {
                "main_title": {
                    "type": "string",
                    "description": "A brief title of the work",
                    "minLength": 10
                },
                "main_introduction": {
                    "type": "string",
                    "description": "An introduction to the work",
                    "minLength": 10
                },
                "sections": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_title": {
                                "type": "string",
                                "description": "A brief title of the section in the work",
                                "minLength": 3
                            },
                            "section_introduction": {
                                "type": "string",
                                "description": "Introduction to the section in the work",
                                "minLength": 10
                            },
                            "subsections": {
                                "type": "array",
                                "minItems": 2,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subsection_title": {
                                            "type": "string",
                                            "description": "The title of the subsection",
                                            "minLength": 3
                                        },
                                        "subsection_introduction": {
                                            "type": "string",
                                            "description": "Introduction to the subsection",
                                            "minLength": 10
                                        },
                                        "subsection_points": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "description": "Ordered bullet points for the subsection. (do not include the bullet point symbol)",
                                                "minLength": 3
                                            },
                                            "minItems": 1
                                        },
                                        "subsection_conclusion": {
                                            "type": "string",
                                            "description": "A summary conclusion of the subsection",
                                            "minLength": 10
                                        }
                                    },
                                    "required": ["subsection_title", "subsection_introduction", "subsection_points", "subsection_conclusion"]
                                }
                            },
                            "section_conclusion": {
                                "type": "string",
                                "description": "A summary conclusion of the section",
                                "minLength": 10
                            }
                        },
                        "required": ["section_title", "section_introduction", "subsections", "section_conclusion"]
                    }
                },
                "main_conclusion": {
                    "type": "string",
                    "description": "A summary conclusion of the work",
                    "minLength": 10
                }
            },
            "required": ["main_title", "main_introduction", "sections", "main_conclusion"]
        }
    }
}'''

no_context_instruction = "Create a hierarchical outline of the following text snippet, including introductions and conclusions for each section:\n\n"

function_json = json.loads(function_json_string)

json_string = json.dumps(function_json)

#tools_string="[AVAILABLE_TOOLS] ["+json_string+"][/AVAILABLE_TOOLS]"
#instruction_string="[INST] "+instruction+" [/INST]"
#prompt = tools_string+instruction_string + "[TOOL_CALLS]"

function_params=function_json["function"]["parameters"].copy()
function_params["title"] = function_json["function"]["name"]+"_input_params"
function_params_string = json.dumps(function_params)

config = AutoConfig.from_pretrained(MODEL_DIR_PATH)

print("init empty weights")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

print("infer auto device map")

max_memory = {0:"4.5GiB", 1:"4.5GiB","cpu":"0GiB"}


print("max_memory", max_memory)

device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["MistralDecoderLayer"],
        dtype=torch.int8
    )

print(device_map)

print("load checkpoint and dispatch")
model = outlines.models.transformers(
        MODEL_DIR_PATH,
        #device="cuda",   #why... https://github.com/outlines-dev/outlines/blob/a987159860a6dd3a83d2f2376f36ab28ef45decd/outlines/models/transformers.py#L229
        device=None,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "device_map": device_map,
            "load_in_8bit": True,
            "config":config,
        },
    )

def generate_outline_no_prior_context(text_to_outline):
    generator = outlines.generate.json(model, function_params_string)
    tools_string="[AVAILABLE_TOOLS] ["+json_string+"][/AVAILABLE_TOOLS]"
    instruction_string="[INST] "+no_context_instruction + text_to_outline + " [/INST]"
    prompt = tools_string + instruction_string + "[TOOL_CALLS]"
    output = generator(prompt)
    return output

#def generate_outline_with_prior_context(text_to_outline, prior_context):
    

ebook_dir = "/ebooks"
ebook_analysis_dir = "/ebook_analyses"

from langchain_text_splitters import RecursiveCharacterTextSplitter

MAXIMUM_CHUNK_SIZE = 8000
OVERLAP = 500
minimum_chunk_size = MAXIMUM_CHUNK_SIZE // 2

text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "SCENE__BREAK__SCENE__BREAK",
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    chunk_size=MAXIMUM_CHUNK_SIZE,
    chunk_overlap=OVERLAP,
    length_function=len,
    is_separator_regex=False,
)


for ebook in os.listdir(ebook_dir):
    if ebook.endswith(".txt"):
        with open(os.path.join(ebook_dir, ebook), "r") as f:
            text = f.read()
        chunks = [section.page_content for section in text_splitter.create_documents([text])]
        aggregated_chunks = []
        current_aggregate = chunks[0]
        for chunk in chunks[1:]:
            if len(current_aggregate) + len(chunk) < minimum_chunk_size:
                current_aggregate += "\n" + chunk
            else:
                aggregated_chunks.append(current_aggregate.strip().replace("SCENE__BREAK__SCENE__BREAK", ""))
                current_aggregate = chunk
        if current_aggregate:
            aggregated_chunks.append(current_aggregate)
        section_dir = os.path.join(ebook_analysis_dir, ebook.split(".")[0])
        if not os.path.exists(section_dir):
            os.makedirs(section_dir)
        prior_sections = []
        for i, section in enumerate(aggregated_chunks):
            j=str(i).zfill(6)
            with open(os.path.join(section_dir, f"section_{j}.txt"), "w") as f:
                f.write(section)
            outline = generate_outline(section,prior_sections)
            with open(os.path.join(section_dir, f"section_{j}_outline.json"), "w") as f:
                print(outline)
                f.write(str(outline))
            prior_sections.append(section)