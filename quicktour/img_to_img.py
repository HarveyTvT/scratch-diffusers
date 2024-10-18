import torch
from diffusers import AutoPipelineForImage2Image, StableDiffusionLatentUpscalePipeline, StableDiffusionUpscalePipeline
from diffusers.utils import load_image, make_image_grid


def img2img(prompt, init_image, strength=0.5, guidance_scale=7.5, output_path="output/generated_img2img.png"):
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipeline.enable_model_cpu_offload()
    pipeline.enable_xformers_memory_efficient_attention()

    init_img = load_image(init_image)
    image = pipeline(prompt, image=init_img, strength=strength,
                     guidance_scale=guidance_scale).images[0]
    make_image_grid([init_img, image], rows=1, cols=2).save(output_path)


# prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
# img2img(prompt, "./input/cat.png")


prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# img2img(prompt, "./input/astronaut.png", strength=0,
#         output_path="output/generated_img2img_0.png")
# img2img(prompt, "./input/astronaut.png", strength=0.4,
#         output_path="output/generated_img2img_0_4.png")
# img2img(prompt, "./input/astronaut.png", strength=0.6,
#         output_path="output/generated_img2img_0_6.png")
# img2img(prompt, "./input/astronaut.png", strength=1.0,
#         output_path="output/generated_img2img_1_0.png")

# img2img(prompt, "./input/astronaut.png", guidance_scale=0.1,
#         output_path="output/generated_img2img_0_1.png")
# img2img(prompt, "./input/astronaut.png", guidance_scale=5.0,
#         output_path="output/generated_img2img_5.png")
# img2img(prompt, "./input/astronaut.png", guidance_scale=10.0,
#         output_path="output/generated_img2img_10.png")


def img2upscaler2superres(prompt, init_image):
    # 1. image to image
    pipeline = AutoPipelineForImage2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipeline.enable_model_cpu_offload()
    pipeline.enable_xformers_memory_efficient_attention()

    init_img = load_image(init_image)
    latent_image = pipeline(prompt, image=init_img, strength=0.5,
                            guidance_scale=7.5, output_type="latent").images[0]

    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16,
    )
    upscaler.enable_model_cpu_offload()
    upscaler.enable_xformers_memory_efficient_attention()

    upscaled_image = upscaler(
        prompt, image=latent_image, output_type="latent").images[0]

    super_res = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    super_res.enable_model_cpu_offload()
    super_res.enable_xformers_memory_efficient_attention()

    result_image = super_res(prompt, image=upscaled_image).images[0]
    make_image_grid(
        [init_image, result_image.resize((512, 512))], rows=1, cols=2)


img2upscaler2superres(prompt, "./input/astronaut.png")
