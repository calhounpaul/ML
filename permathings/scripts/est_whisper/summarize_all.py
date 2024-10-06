import os
import json
import requests
from pathlib import Path

# Configuration
THIS_DOCKER_CONTAINER_NAME = "transformers-whisper-est"

# Construct paths
THIS_DIR_PATH = Path(__file__).parent
SCRIPTS_DIR_PATH = THIS_DIR_PATH.parent
PERMATHINGS_DIR_PATH = SCRIPTS_DIR_PATH.parent
ML_PROJECT_DIR_PATH = PERMATHINGS_DIR_PATH.parent
EPHEMERA_DIR_PATH = ML_PROJECT_DIR_PATH / "ephemera"
SHARED_CACHES_DIR_PATH = EPHEMERA_DIR_PATH / "shared_caches"
EPHEMERAL_CACHE_PATH = EPHEMERA_DIR_PATH / THIS_DOCKER_CONTAINER_NAME
LIBRARIES_DIR_PATH = PERMATHINGS_DIR_PATH / "libs"

TRANSCRIBED_DIR = EPHEMERAL_CACHE_PATH / "transcribed_episodes"
SUMMARIZED_DIR = EPHEMERAL_CACHE_PATH / "summarized_episodes"

# Create summarized_episodes directory if it doesn't exist
SUMMARIZED_DIR.mkdir(parents=True, exist_ok=True)

FEW_SHOT_EXAMPLE = [{
  "role": "user",
  "content": "Summarize the following Estonian transcription in English:\nnoh, see \"Pulp Fiction\" on ikka üks pöördeline film. Tarantino lõi sellega täiesti uue standardi 90ndate kinole.\nmäletan kui esimest korda nägin, olin täiesti šokis sellest ajajoone hüppamisest. alguses ei saanud üldse aru, mis toimub.\njaa, see kronoloogia lõhkumine oli tõesti geniaalne. ja need tegelased - Uma Thurmani Mia Wallace, John Travolta Vincent Vega rollis. kuidagi nii elutruu ja samas täiesti jabur seltskond.\nmulle jäi eriti meelde see stseen, kus nad lähevad sinna retro-restorani ja tantsivad. see twist contest. nii lahe muusika ja koreograafia.\naga see overdose'i stseen oli küll hull. kui nad torkasid talle seda nõela südamesse, ma olin täiesti hirmul.\njah, Tarantino oskab ikka vaatajat šokeerida. aga samas on seal ka nii palju huumorit ja popkultuuri viiteid.\nnäiteks see \"royale with cheese\" dialoog autos. nii tavaline jutuajamine ja samas nii meeldejääv.\nmul on siiamaani see \"Misirlou\" lugu peas, kui sellele filmile mõtlen. perfektne soundtrack.\naga jah, üks parimaid 90ndate filme kindlasti. muutis täielikult seda, kuidas lugusid kinolinal jutustatakse.\n\nNow, respond only with an English summary of the Estonian podcast transcript."
},
{
  "role": "assistant",
  "content": "The podcast hosts are discussing the movie Pulp Fiction. They praise the film as groundbreaking, noting Tarantino's innovative approach to storytelling, particularly the non-linear timeline. They highlight memorable scenes such as the twist contest dance and the adrenaline shot sequence. The conversation touches on the film's impact on 90s cinema, its blend of shock value and humor, iconic characters, and unforgettable dialogue like the \"royale with cheese\" scene. The speakers also mention the excellent soundtrack, especially \"Misirlou.\" Overall, they regard Pulp Fiction as one of the best films of the 1990s, crediting it with changing the way stories are told in cinema."
}]

def summarize_chunk(chunk: str, chunk_type: str) -> str:
    api_data = {
        "model": "qwen2.5:32b",
        "messages": FEW_SHOT_EXAMPLE + [
            {
                "role": "user",
                "content": f"Excellent summary. Please perform the exact same task for the following {chunk_type} of an Estonian podcast transcript:\n{chunk}\n\nNow, respond only with an English summary of this {chunk_type} of the Estonian podcast transcript."
            }
        ],
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/chat", json=api_data)
        response.raise_for_status()
        full_response = response.json()
        summary = full_response.get('message', {}).get('content')
        
        if not summary:
            raise ValueError("Summary is empty or null")
        
        return summary
    except Exception as e:
        print(f"Error summarizing {chunk_type}: {e}")
        return f"Error summarizing {chunk_type}: {str(e)}"

def summarize_transcript(input_file: Path, output_file: Path):
    print(f"Summarizing file: {input_file}")
    
    with input_file.open('r', encoding='utf-8') as f:
        transcript_content = f.read()
    
    total_length = len(transcript_content)
    chunk1 = transcript_content[:int(total_length * 0.2)]
    chunk2 = transcript_content[int(total_length * 0.2):int(total_length * 0.6)]
    chunk3 = transcript_content[int(total_length * 0.6):]

    summary1 = summarize_chunk(chunk1, "first 20%")
    summary2 = summarize_chunk(chunk2, "middle 40%")
    summary3 = summarize_chunk(chunk3, "last 40%")

    final_summary = summarize_full_transcript(transcript_content, summary1, summary2, summary3)

    with output_file.open('w', encoding='utf-8') as f:
        f.write(final_summary)
    print(f"Final summary saved to: {output_file}")

def summarize_full_transcript(full_transcript: str, summary1: str, summary2: str, summary3: str) -> str:
    api_data = {
        "model": "qwen2.5:32b",
        "messages": FEW_SHOT_EXAMPLE + [
            {
                "role": "user",
                "content": f"Excellent summary. For your next task, please provide a comprehensive summary of an entire podcast episode based on the following full transcript, followed by chunk summaries:\n\nFull transcript:\n{full_transcript}\n\nSummary of first 20% of the above transcript:\n{summary1}\n\nSummary of middle 40%:\n{summary2}\n\nSummary of last 40%:\n{summary3}\n\nNow, respond only with a complete English summary of the whole podcast episode above."
            }
        ],
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/chat", json=api_data)
        response.raise_for_status()
        full_response = response.json()
        final_summary = full_response.get('message', {}).get('content')
        
        if not final_summary:
            raise ValueError("Final summary is empty or null")
        
        return final_summary
    except Exception as e:
        print(f"Error creating final summary: {e}")
        return f"Error creating final summary: {str(e)}"

def main():
    for input_file in TRANSCRIBED_DIR.glob('*.txt'):
        output_file = SUMMARIZED_DIR / f"{input_file.stem}.sum.txt"
        
        if not output_file.exists():
            summarize_transcript(input_file, output_file)
        else:
            print(f"Summary already exists for: {input_file}")

    print("Summarization process completed.")

if __name__ == "__main__":
    main()