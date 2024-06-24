import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image, make_image_grid


def img2img(prompt, init_image):
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipeline.enable_model_cpu_offload()
    pipeline.enable_xformers_memory_efficient_attention()

    init_img = load_image(init_image)
    image = pipeline(prompt, image=init_img).images[0]
    make_image_grid([init_img, image], rows=1, cols=2).save(
        "output/generated_img2img.png")


prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
img2img(prompt, "./input/cat.png")
