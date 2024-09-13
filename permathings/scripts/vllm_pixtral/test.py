import base64
import requests
from openai import OpenAI



# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


models = client.models.list()

model = models.data[0].id

# Single-image input inference

#image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


## Use base64 encoded image in the payload

file_path = "./objects.jpg"
with open(file_path, "rb") as f:
    image_data = f.read()

image_base64 = base64.b64encode(image_data).decode('utf-8')

## Use image url in the payload

chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What’s in this image? Describe everything you see."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            },
        ],
    }],
    model=model,
    max_tokens=600,
)

result = chat_completion_from_url.choices[0].message.content

print("Chat completion output:", result)

with open("../../persistent_output_examples/pixtral_inference.txt", "w") as f:
    f.write(result)
