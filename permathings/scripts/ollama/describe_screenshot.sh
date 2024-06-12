THIS_MODEL="llava:34b-v1.6"

#THIS_MODEL="llava-phi3"

json_string="{'model': '$THIS_MODEL', 'prompt': 'What is in this picture?', 'stream': False, 'images': [base64.b64encode(open('"screenshot.png"', 'rb').read()).decode('utf-8')]}).json()"
echo $json_string
command_string="import requests, base64, json; print(json.dumps(requests.post('http://localhost:11434/api/generate', json="$json_string", indent=2))"
response=$(python3 -c "$command_string")
echo $response | jq . > screenshot.png.json
