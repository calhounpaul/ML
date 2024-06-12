#/ephemeral_cache/mistral-inst-v03
#mistralai/Mistral-7B-Instruct-v0.3
import os
import sys
import json

from huggingface_hub import snapshot_download

#if /ephemeral_cache/mistral-inst-v03 does not exist, mkdir
if not os.path.exists("/ephemeral_cache/mistral-inst-v03"):
    os.mkdir("/ephemeral_cache/mistral-inst-v03")

HF_TOKEN = os.getenv("HF_TOKEN")

snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3",local_dir="/ephemeral_cache/mistral-inst-v03", local_dir_use_symlinks=False,use_auth_token=HF_TOKEN,ignore_patterns=["consolidated.safetensors"]) #,local_files_only=True)