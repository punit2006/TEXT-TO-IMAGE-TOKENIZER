import torch
from diffusers import StableDiffusionPipeline
from .config import SD_MODEL, DEVICE

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL)
pipe = pipe.to(DEVICE)

def generate_image_from_text(prompts):
    """Generate images using Stable Diffusion from text prompts"""
    images = []
    for prompt in prompts:
        image = pipe(prompt).images[0]
        images.append(image)
    return images
