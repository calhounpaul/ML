from pydantic import BaseModel
import json
from outlines import models, generate

MODEL_DIR_PATH="/ephemeral_cache/mistral-inst-v03"
#TOKENIZER_PATH=MODEL_DIR_PATH+"/tokenizer.model.v3"

import outlines

function_json_string = '''{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location."
                }
            },
            "required": ["location", "format"]
        }
    }
}'''

instruction = "What is the weather like today in San Francisco?"

function_json = json.loads(function_json_string)

json_string = json.dumps(function_json)

tools_string="[AVAILABLE_TOOLS] ["+json_string+"][/AVAILABLE_TOOLS]"
instruction_string="[INST] "+instruction+" [/INST]"
prompt = tools_string+instruction_string + "[TOOL_CALLS]"

function_params=function_json["function"]["parameters"].copy()
function_params["title"] = function_json["function"]["name"]+"_input_params"
function_params_string = json.dumps(function_params)

model = outlines.models.transformers(MODEL_DIR_PATH)
generator = outlines.generate.json(model, function_params_string)
character = generator(prompt)

print(character)