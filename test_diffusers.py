from diffusers import StableDiffusionPipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained Stable Diffusion model
model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)  # Use GPU for faster inference

# Define your prompt
prompt = "A futuristic cityscape at sunset, cyberpunk style, vibrant colors"

# Generate an image
image = pipe(prompt).images[0]

# Save the generated image
image.save("generated_image.png")
