from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained(
    'anton-l/ddpm-butterflies-128').to("cuda")
image = generator().images[0]
image.save("output/generated.png")
