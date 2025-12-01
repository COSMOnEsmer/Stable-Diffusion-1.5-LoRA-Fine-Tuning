from diffusers import StableDiffusionPipeline

# Load base model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Load LoRA weights
pipe.load_lora_weights("./out/pytorch_lora_weights.safetensors")

# Save the full pipeline locally
pipe.save_pretrained("./sd_with_lora")