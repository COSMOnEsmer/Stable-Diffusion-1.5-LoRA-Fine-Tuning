from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("./sd_with_lora")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

prompt = "A fantasy landscape with mountains and a river at sunset"

# Generate the image
image = pipe(
    prompt,
    num_inference_steps=35,  
    #guidance_scale=7.5,      
    height=512,
    width=512
).images[0]

image.show()
image.save("generated_image.png")