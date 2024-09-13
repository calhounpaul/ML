# pip install huggingface-hub

from huggingface_hub import snapshot_download

snapshot_download(repo_id="mistral-community/pixtral-12b-240910", local_dir="pixtral-12b-240910")
