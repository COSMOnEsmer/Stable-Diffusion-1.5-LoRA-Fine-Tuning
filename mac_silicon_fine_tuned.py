import torch
from diffusers import StableDiffusionPipeline

device = "mps"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

pipe.to(device)

pipe.load_lora_weights(
    "pytorch_lora_weights.safetensors", 
    weight_name=None, 
    adapter_name="default"
)

prompt = "A fantasy landscape with mountains and a river at sunset"

image = pipe(
    prompt,
    num_inference_steps=35,
    height=512,
    width=512,
).images[0]

image.save("output.png")
print("saved!")