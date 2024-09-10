import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import ffmpeg

#prompts by Claude 3.5
prompts = [
  "In the misty depths of Loch Ness, the legendary Nessie gracefully glides through the dark waters. Her long, serpentine neck arches above the surface, revealing glimpses of glistening, prehistoric scales. As she moves, ripples spread across the loch, disturbing the reflections of the ancient Scottish highlands. The creature's eyes, wise and ancient, scan the shoreline, a living relic from a time long past.",

  "The elusive Yeti trudges through the snow-capped peaks of the Himalayas, leaving behind massive footprints that quickly fill with fresh powder. Its shaggy white fur blends seamlessly with the surrounding snow, making it nearly invisible against the stark landscape. The creature's breath forms clouds in the frigid air as it surveys its domain, a living legend of the world's highest mountains.",

  "In the hidden valley of Shangri-La, time stands still. Lush gardens bloom with otherworldly flowers, their petals shimmering with an inner light. Ageless sages meditate beneath ancient trees, their faces serene and unlined despite centuries of contemplation. A soft, ethereal music fills the air, emanating from nowhere and everywhere at once. Here, in this mythical paradise, the secrets of eternal youth and profound wisdom are whispered on the wind.",

  "A peculiar creature, part rabbit and part antelope, sniffs the air in the golden prairie light. Its long, fur-covered rabbit ears stand erect, swiveling independently between its tall rigid antlers. The jackalope's compact, furry body crouches low, supported by powerful hind legs typical of a hare. Its twitching nose and whiskers betray its lagomorph nature, while small, branching antlers sprout incongruously from its head. For a brief moment, this impossible hybrid embodies the fantastical folklore of the American West.",
]

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

for prompt in prompts:
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=24,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    filename = "".join([c if c.isalnum() else "_" for c in prompt[:30].lower()]) + ".mp4"
    print(f"Exporting video to {filename}")

    export_to_video(video, filename, fps=8)

    stream = ffmpeg.input(filename)
    gif_filename = filename.replace(".mp4", ".gif")
    stream = ffmpeg.output(stream, gif_filename)
    ffmpeg.run(stream)
