import torch
from diffusers import StableDiffusionPipeline

if torch.cuda.is_available():
    device = "cuda"
    weight_dtype = torch.float16  # Nvidia GPUs - float16
elif torch.backends.mps.is_available():
    device = "mps"
    weight_dtype = torch.float16  # Apple Silicon - float16
else:
    device = "cpu"
    weight_dtype = torch.float32  # CPU - float32 

print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=weight_dtype
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