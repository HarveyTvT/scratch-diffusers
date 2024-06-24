from diffusers import AutoPipelineForText2Image, ControlNetModel
from diffusers.utils import load_image
import torch


def sd15(prompt, negative_prompt="", width=512, height=512, guidance_scale=7.5, output="output/generated_sd15.png"):
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16").to("cuda")
    image = pipeline(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height,
                     guidance_scale=guidance_scale).images[0]
    image.save(output)


def sdxl(prompt):
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16").to("cuda")
    image = pipeline(prompt).images[0]
    image.save("output/generated_sdxl.png")


def sd15_controlnet(prompt, poseImage):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16, variant="fp16").to("cuda")
    pose_image = load_image(poseImage)

    pipeline = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16").to("cuda")
    generator = torch.Generator("cuda").manual_seed(31)
    image = pipeline(prompt, image=pose_image, generator=generator).images[0]
    image.save("output/generated_sd15_controlnet.png")


# prompt = "stained glass of darth vader, backlight, centered composition, masterpiece, photorealistic, 8k"
# sd15(prompt)
# sdxl(prompt)


prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# sd15(prompt)
# sdxl(prompt)
# sd15_controlnet(prompt, "./input/images_control.png")

# sd15(prompt, width=512, height=768)

# sd15(prompt, guidance_scale=2.5, output="output/generated_sd15_2_5.png")
# sd15(prompt, guidance_scale=7.5, output="output/generated_sd15_7_5.png")
# sd15(prompt, guidance_scale=10.5, output="output/generated_sd15_10_5.png")

negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"
sd15(prompt=prompt, negative_prompt=negative_prompt)
