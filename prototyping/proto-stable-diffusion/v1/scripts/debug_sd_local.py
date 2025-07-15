"""
Takeaways:
1. Even locally running SD models can create surprisingly good generic images.
2. When prompted with domain specific requests, they are terrible.
3. Tried a lot of things; even incorporated mixtral to generate a meta-prompt using the concept.
   But, locally running SD even with the meta-prompt is just not good enough.
   Must fall back on Dall-E-2 using OpenAI API keys. 
"""
import os
import ipdb
import torch
from diffusers import StableDiffusionXLPipeline
import subprocess

from time import time

GENERATE = False

init = time()

# Load the pipeline once globally.
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("mps")

fin1 = time()
print(f"Time to load model:{fin1 - init:.2f}s.")

#prompt = "A futuristic cityscape at sunset"
#image = pipe(prompt).images[0]
#image.save("output.png")

def generate_image_local(prompt, output_path="generated_image.png"):
    """
    Generates an image locally using Hugging Face's diffusers Stable Diffusion pipeline.
    """
    image = pipe(prompt).images[0]
    image.save(os.path.join("./data/tmp/", output_path))
    print(f"Image saved to {output_path}")

def generate_image_prompt(concept, model="mixtral"):

   meta_prompt = f"""
You are an expert at writing prompts for AI image generation using Stable Diffusion.

Given a technical concept, generate a **clear, concise visual prompt** in less than 70 words that issuitable for a Stable Diffusion model.

Requirements:
- The image should be a **2D technical diagram** (no 3D visuals).
- Use short, descriptive phrases — **no long paragraphs**.
- Focus on spatial layout, visual elements, color/style.
- Avoid technical jargon or implementation details.
- Include only essential mathematical notation, and only if it helps the visual design.
- Use stylistic cues like "flat", "vector-style", "pastel", or "minimalist".
- **Output only the final prompt** (no preamble or explanation).

Concept: {concept}
"""

   result = subprocess.run(
       ["ollama", "run", model],
       input=meta_prompt.encode(),
       capture_output=True,
       timeout=120
   )
   return result.stdout.decode("utf-8").strip()

#fin2 = time()
#prompt4 = "diagram or illustration of an encoder and decoder stack"
#prompt4_llm = generate_image_prompt(prompt4)
#fin3 = time()
#print(f"Time to generate prompt for SD: {fin3 - fin2:.2f}s.")
# print(f"\n\n{prompt3}: {prompt3_llm}\n\n")
# print(f"\n\n{prompt4}: {prompt4_llm}\n\n")
# generate_image_local(prompt3_llm, output_path="prompt3.png")

#ipdb.set_trace()

prompt = "modern professional slide background, minimal design, blue tones"
image = pipe(prompt,
             negative_prompt="low quality, blurry, text, watermark",
             height=1080,
             width=1920,
             num_inference_steps=100,
             guidance_scale=10,).images[0]
image.save("slide_background.png")


# prompt = "A fantasy forest with magical creatures"
# image = pipe(prompt).images[0]
# image.save("output.png")


# if GENERATE:
#    prompt1 = "A futuristic cityscape at sunset"
#    prompt2 = prompt1 + ", in the style of a vector image."
#
#    generate_image_local(prompt1, output_path="image1.png")
#    generate_image_local(prompt2, output_path="image2.png")
#
# if GENERATE:
#    prompt3 = "diagram or illustration of scaled dot-product attention"
#    generate_image_local(prompt3, output_path="image3a.png")
#
# if GENERATE:
#    prompt4 = "diagram or illustration of an encoder and decoder stack"
#    generate_image_local(prompt4, output_path="image4a.png")
#
# if GENERATE:
#    prompt5 = "A labeled diagram showing the scaled dot-product attention mechanism, with arrows pointing from query to key and softmax applied, matrix multiplications and attention scores labeled, clean white background. vector infographic, white background"
#    prompt6 = "A clean technical diagram showing an encoder-decoder architecture. The encoder consists of stacked blocks labeled with attention and feed-forward layers. The decoder mirrors this with cross-attention. Arrows connect the stacks. White background, vector art style."
#    prompt7 = (
#        "Technical diagram of scaled dot-product attention: "
#        "Query, Key, and Value matrices shown as rectangular blocks; arrows showing dot product and softmax operation; "
#        "attention scores applied to Value; clean lines; vector-style; white background"
#    )
#    prompt8 = (
#        "Infographic of transformer encoder-decoder stack: left column with encoder blocks, "
#        "right column with decoder blocks, arrows showing flow, labeled layers like self-attention and feed-forward, "
#        "minimalist design, vector-style"
#    )
#
#    generate_image_local(prompt5, output_path="image5.png")
#    generate_image_local(prompt6, output_path="image6.png")
#    generate_image_local(prompt7, output_path="image7.png")
#    generate_image_local(prompt8, output_path="image8.png")

#    meta_prompt_vn2 = f"""
# You are an expert in visual design for technical diagrams.
#
# Given a technical concept, write a detailed visual prompt suitable for a Stable Diffusion image generation model.
# Use visual language, spatial layout, color/styling notes, and avoid jargon. Output only the prompt text.
#
# Concept: {concept}
# """
#
#    meta_prompt_vn1 = f"""
# You are an expert in visual design prompts for AI image generation.
#
# Given a technical concept, write a **concise visual prompt** for a Stable Diffusion model.
#
# Use:
# - Simple visual language
# - Style and layout suggestions (brief)
# - No long paragraphs
# - No jargon or equations unless visually relevant
#
# Output only the final prompt as one paragraph.
#
# Concept: {concept}
# """
