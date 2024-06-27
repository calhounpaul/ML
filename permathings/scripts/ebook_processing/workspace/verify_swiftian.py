import os, sys, re, json, random, time
from unidecode import unidecode

#cuda visible devices= 0,1
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pydantic import BaseModel
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, logging
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
torch.cuda.empty_cache()
import outlines


MODEL_DIR_PATH="/ephemeral_cache/mistral-inst-v03"
LOAD_MODEL = True


ebook_dir = "/ebooks"
ebook_analysis_dir = "/ebook_analyses"
swift_quotes_dir = ebook_analysis_dir+"/swift_quotes"
if not os.path.exists(swift_quotes_dir):
    os.makedirs(swift_quotes_dir)
jswift_book = "jswift.txt"
jswift_path = os.path.join(ebook_dir, jswift_book)
jswift_string = open(jswift_path).read()

scene_split_lines = jswift_string.split("SCENE__BREAK__SCENE__BREAK")

reduced_lines = []
for line in scene_split_lines:
    if len(line) < 6000:
        continue
    line = line[2000:-2000]
    if line.count("\n")<4:
        continue
    line_split_lines = line.split("\n")[1:-1]
    for line in line_split_lines:
        if len(line) < 800:
            continue
        skip=False
        for dropstring in ["Swift","[T.S."]:
            if dropstring in line:
                skip=True
                break
        if skip:
            continue
        if line.startswith("Page ") and line.split(" ")[1].endswith("."):
            continue
        sentences = []
        next_sentence = line.split(".")[0] + "."
        for sentence in line.split(".")[:-1]:
            this_sentence = next_sentence + " " + sentence + "."
            
        reduced_lines.append(line.strip())

from sentence_splitter import SentenceSplitter, split_text_into_sentences

if not os.path.exists(ebook_analysis_dir + "/jswift_chunks_data.json"):
    splitter = SentenceSplitter(language='en')
    #print(splitter.split(text='This is a paragraph. It contains several sentences. "But why," you ask?'))

    reduced_lines = list(set(reduced_lines))

    chunks_and_sentences = []

    for line in reduced_lines:
        sentences = splitter.split(text=line)
        data_dict = {
            "chunk": line,
            "sentences": sentences.copy(),
            "starting_position": jswift_string.find(line),
        }
        chunks_and_sentences.append(data_dict)
        
    with open(ebook_analysis_dir + "/jswift_chunks_data.json", "w") as f:
        json.dump(chunks_and_sentences, f, indent=4)
else:
    with open(ebook_analysis_dir + "/jswift_chunks_data.json", "r") as f:
        chunks_and_sentences = json.load(f)


main_description = "Stores input normalized restatements/translations of sentences from a paragraph of text written by author and playwright Jonathan Swift."

function_json_string = '''{
    "type": "function",
    "function": {
        "name": "store_normalized_swiftian_text",
        "description": "''' + main_description + '''",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}'''

#no_context_instruction = "Normalize the following paragraph of text written by Jonathan Swift:"

function_json = json.loads(function_json_string)

json_string = json.dumps(function_json)

#print(json_string)

#input()

#function_params=function_json["function"]["parameters"].copy()
#function_params["title"] = function_json["function"]["name"]+"_input_params"
#function_params_string = json.dumps(function_params)

config = AutoConfig.from_pretrained(MODEL_DIR_PATH)
"""
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
"""
if LOAD_MODEL:
    print("load checkpoint and dispatch")
    model = outlines.models.transformers(
            MODEL_DIR_PATH,
            device="cuda",   #why... https://github.com/outlines-dev/outlines/blob/a987159860a6dd3a83d2f2376f36ab28ef45decd/outlines/models/transformers.py#L229
            #device=None,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                #"device_map": device_map,
                #"load_in_8bit": True,
                "config":config,
            },
        )

#def generate_outline_no_prior_context(text_to_outline):
#    generator = outlines.generate.json(model, function_params_string)
#    tools_string="[AVAILABLE_TOOLS] ["+json_string+"][/AVAILABLE_TOOLS]"
#    instruction_string="[INST] "+no_context_instruction + text_to_outline + " [/INST]"
#    prompt = tools_string + instruction_string + "[TOOL_CALLS]"
#    output = generator(prompt)
#    return output


def validate_file(existing_file):
    if "output" in existing_file:
        skip = True
        for sentence_key in existing_file["output"]:
            output_string = existing_file["output"][sentence_key]
            #redo "}", "{","[","]","<",">","`","~","@","#","^","*"
            if "normalized" in output_string or len(output_string) < 6 or any([c in output_string for c in ["}", "{","[","]","<",">","`","~","@","#","^","*","swiftian","_"]]):
                skip=False
        if skip:
            #if all sentences are valid, skip
            return True
    return False

PRE_POST_CONTEXT_SIZE = 1500

for k, chunk in enumerate(chunks_and_sentences):
    chunk["num"]=k
    chunks_and_sentences[k]=chunk

random.shuffle(chunks_and_sentences)

start_time = time.time()
count=0

for chunk in chunks_and_sentences:
    try:
        k = chunk["num"]
        save_path = os.path.join(swift_quotes_dir, "chunk_"+str(k).zfill(6)+".json")
        existing_file = None
        if os.path.exists(save_path):
            if os.path.getsize(save_path) > 10: # and unidecode(chunk["chunk"]) in open(save_path).read():
                with open(save_path, "r") as f:
                    existing_file = json.load(f)
                    print("found existing file", save_path)
        if existing_file is not None:
            if validate_file(existing_file):
                print("skipping", save_path)
                continue
        count+=1
        print("Processing chunk", k)
        print("Average time per chunk:", (time.time()-start_time)/count)
        chunk_sentences = chunk["sentences"]
        #print(chunk_sentences)
        #input()
        chunk_start = chunk["starting_position"]
        chunk_string = chunk["chunk"]
        chunk_prior_context = jswift_string[chunk_start-PRE_POST_CONTEXT_SIZE:chunk_start]
        chunk_post_context = jswift_string[chunk_start+len(chunk_string):chunk_start+len(chunk_string)+PRE_POST_CONTEXT_SIZE]
        this_function_json = function_json.copy()
        current_params = {
                "type": "object",
                "properties": {},
                "required": [],
            }
        #any sentence that is less than 10 characters long should be combined with the next sentence and the next sentence should be removed
        fixed_chunk_sentences = []
        for i, sentence in enumerate(chunk_sentences):
            if len(sentence) < 20:
                if i+1 < len(chunk_sentences):
                    chunk_sentences[i+1] = sentence + " " + chunk_sentences[i+1]
            else:
                fixed_chunk_sentences.append(sentence)
        chunk_sentences = fixed_chunk_sentences
        for i, sentence in enumerate(chunk_sentences):
            first_twenty_chars = [c if c.isalpha() else "_" for c in sentence[:20]]
            #property_name = "sentence-" + str(i) + "_" + "".join(first_twenty_chars).rstrip("_")
            property_name = "normalized_sentence-" + str(i).zfill(3)
            truncd_sentence=sentence
            if len(sentence) > 50:
                truncd_sentence = sentence[:25] + "..." + sentence[-25:]
            current_params["properties"][property_name] = {
                "type": "string",
                # Exclude " + '"}", "{","[","]","<",">","`","~","@","#","^","*","swiftian","_",' + " and ensure the sentence is at least 75% of the original length.
                "description": "A modernized/normalized translation of sentence #" + str(i) + ": '" + unidecode(truncd_sentence) + "'. Do not include any of the following characters: " + '"}", "{","[","]","<",">","`","~","@","#","^","*","_",' + ".",
                #regex should exclude "_"
                #minimum length should be 0.75*len(sentence)
            }
            current_params["required"].append(property_name)
        this_function_json["function"]["parameters"] = current_params
        given_info_as_json_dict = {
            "prior_context": "..." + unidecode(chunk_prior_context),
            "chunk": unidecode(chunk_string),
            "post_context": unidecode(chunk_post_context) + "...",
            "chunk_sentences_to_normalize":{
                "sentence-" + str(i).zfill(3): unidecode(chunk_sentences[i]) for i in range(len(chunk_sentences))
            },
            #"chunk_sentences_to_normalize": chunk_sentences,
        }
        #print(json.dumps(this_function_json, indent=4))
        #input()
        #instruction = """Use the function "store_normalized_swiftian_text" to save a plain english restatement/translation of each sentence from the following quote by 18th century Irish Playwright Jonathan Swift. Make sure to preserve the pov, tone, and meaning of the original text. Simplify archaic vocabulary by replacing old-fashioned words with their modern equivalents, such as changing "hath" to "has" and "thou" to "you". Update grammar and syntax by adjusting the sentence structure to match contemporary grammar rules, modifying word order, and punctuation for clarity, such as changing "The streets wore a desolate aspect" to "The streets looked deserted". Clarify obsolete references by providing context or simplifying historical events, cultural practices, or idiomatic expressions that are no longer commonly understood. Maintain the original meaning by ensuring the essence, tone, and intent of Swift's writing are preserved without altering the original message. Remove redundancies and formalities by streamlining overly formal expressions and eliminating repetitive phrases, such as shortening long-winded sentences and removing unnecessary words. For example, convert Swift's text from "Gulliver's Travels": "As I walked through the suburbs to the city, I observed that the shops and houses were shut, and the streets wore a desolate aspect. But in the great square, there was a multitude of people in the utmost confusion, and the man in the moon was just setting," to "As I walked through the outskirts of the city, I noticed that the shops and houses were closed, and the streets looked deserted. But in the main square, there was a large, chaotic crowd, and the moon was just setting." By following these instructions as well as your own brilliant intuition, you can make Swift's prose more accessible to modern readers while retaining the richness and depth of the original narrative. Here is the data, including the specific sentences to normalize:
        instruction = """Store a normalized/plain English restatement/translation of each sentence from the following quote by 18th century Irish Playwright Jonathan Swift. Make Swift's prose more accessible to modern readers while retaining the meaning and depth of the original narrative. Here is the data, including the specific sentences to normalize:

    """ + json.dumps(given_info_as_json_dict, indent=4)
        function_params=function_json["function"]["parameters"].copy()
        function_params["title"] = function_json["function"]["name"] + "_input_params"
        this_function_string = json.dumps(this_function_json)
        for prop_name in function_params["properties"]:
            #add "regex": "^[^{}<>`~@#^*]*$",
            #also "_" should be excluded
            #so in the end the regex will be "^[^{}<>`~@#^*_]*$"
            #function_params["properties"][prop_name]["pattern"] = r"^[^{}<>`~@#^*_]*$"
            #"minLength": int(0.5*len(sentence)),
            #"maxLength": 1000,
            sentence = chunk_sentences[int(prop_name.split("-")[-1])]
            function_params["properties"][prop_name]["minLength"] = int(0.5*len(sentence))
            function_params["properties"][prop_name]["maxLength"] = int(1.5*len(sentence))
        function_params_string = json.dumps(function_params)
        generator = outlines.generate.json(model, function_params_string)
        tools_string="[AVAILABLE_TOOLS] ["+this_function_string+"][/AVAILABLE_TOOLS]"
        instruction_string="[INST] "+ instruction + " [/INST]"
        prompt = tools_string + instruction_string + "[TOOL_CALLS]"
        output = generator(prompt)
        try:
            output_json = json.loads(output)
        except:
            output_json = output
        data_to_save = {
            "prompt_string": prompt,
            "given_info_as_json_dict": given_info_as_json_dict,
            "function": this_function_json,
            "output": output,
            "output_json": output_json,
        }
        with open(save_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        print("Saved chunk", k, "to", save_path)
    except Exception as e:
        #if keyboard interrupt, save progress
        if "KeyboardInterrupt" in str(e):
            break
        print(e)
        print("Error processing chunk", k)