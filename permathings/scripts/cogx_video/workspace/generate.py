import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

prompt = "A panda appears on a dating gameshow and is ignored by the other contestants."

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

filename = "".join([c if c.isalnum() else "_" for c in prompt[:30].lower()]) + ".mp4"
print(f"Exporting video to {filename}")

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, filename, fps=8)
