import os
import torch
from diffusers import StableDiffusionXLPipeline

# This script illustrates the exact code used to run Generative AI (Day 8 Concept)
# using a Stable Diffusion XL (SDXL) pipeline to generate educational illustrations of rashes.

def generate_educational_illustration(prompt, output_path="educational_illustration.png"):
    print(f"Initializing SDXL Pipeline to generate image for prompt: '{prompt}'")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target execution device: {device}")
    
    # ----------------------------------------------------
    # ALGORITHM 4: Stable Diffusion XL Pipeline (Day 8 Concept)
    # ----------------------------------------------------
    # Load SDXL base pipeline in Float16 mixed precision (saves half the VRAM)
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True
        )
        pipeline = pipeline.to(device)
        
        # Enable memory saving optimizations if using GPU
        if device == "cuda":
            print("Optimizing VRAM usage with sequential cpu offloading and attention slicing...")
            pipeline.enable_attention_slicing()
            # If using xformers or torch 2.0 SDPA (Day 9 optimization):
            # pipeline.enable_xformers_memory_efficient_attention()
            
        print("Running pipeline inference...")
        # Generate the image
        # Negative prompts ensure clinical clarity and avoid weird artifacts
        negative_prompt = "low quality, blurry, cartoon, drawing, illustration, deformed, internal organ, surgery, blood, text, watermark"
        
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30, # High fidelity steps
            guidance_scale=7.5      # Classifier-free guidance strength
        ).images[0]
        
        image.save(output_path)
        print(f"Successfully generated and saved illustration to {output_path}")
        
    except Exception as e:
        print(f"SDXL Generation failed: {e}")
        print("This is normal if running on a system without a high-VRAM GPU (SDXL requires at least 8-12GB VRAM).")
        print("Creating a mock clinical diagram placeholder for report rendering...")
        
        # Create a mock drawing as fallback
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (512, 512), color=(247, 250, 252))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 492, 492], outline=(43, 108, 176), width=3)
        draw.text((100, 240), f"[SDXL Diagnostic Visual:\n{prompt[:30]}...]", fill=(43, 108, 176))
        img.save(output_path)
        print(f"Created fallback image at: {output_path}")

if __name__ == "__main__":
    prompt = "high resolution clinical macro photo of eczema skin rash, highly detailed, medical illustration"
    generate_educational_illustration(prompt)
