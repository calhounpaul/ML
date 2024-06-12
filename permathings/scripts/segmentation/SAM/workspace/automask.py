import numpy as np
import matplotlib.pyplot as plt
import gc

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks,output_path="out.png"):
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for mask in masks:
        show_mask(mask, ax=ax, random_color=True)
    plt.axis("off")
    plt.show()
    #export as png
    plt.savefig(output_path)
    del mask
    gc.collect()

from transformers import pipeline
generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=0)

from PIL import Image
import requests

img_file_path = "screenshot.png"

#img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

#raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
raw_image = Image.open(img_file_path).convert("RGB")

#plt.imshow(raw_image)

outputs = generator(raw_image, points_per_batch=64)

masks = outputs["masks"]
print(masks)
#show_masks_on_image(raw_image, masks)
plt.imshow(np.array(raw_image))
ax = plt.gca()
ax.set_autoscale_on(False)
for mask in masks:
    #show_mask(mask, ax=ax, random_color=True)
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    #del mask
plt.axis("off")
#plt.show()
plt.savefig("out.png")
gc.collect()