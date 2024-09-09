import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import ffmpeg

prompts = [
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
