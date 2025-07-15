"""
TAKEAWAYS:
1. Locally run Stable Diffusion pipelines are terrible.
2. Free APIs don't work: Replicate, HuggingFace, OpenRouter, DeepAI.
    a. OpenAI on paid mode is the only way.
3. Dall-E-2 is good enough since PPTs don't need high fidelity information.
"""
import requests
import ipdb
import openai
import torch

from diffusers import StableDiffusionXLPipeline
from utils.memory_management import cleanup

def generate_image_local(pos_prompt, neg_prompt, mode="general", output_path="generated_image.png", num_inference_steps=100):
    """
    Generates an image locally using Hugging Face's diffusers StableDiffusionXLPipeline.
    guidance_scale is set to 12 for better alignment with the text prompt.

    Args:
        prompt (str): The text prompt to guide the image generation process.
        output_path (str, optional): The file path where the generated image will be saved.
            Defaults to "generated_image.png".
    Returns:
        str: The file path where the generated image is saved.
    Raises:
        ValueError: If the prompt is empty or invalid.
        OSError: If there is an issue saving the generated image to the specified output path.
    """
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("mps")

    if mode=="general":
        image = pipe(prompt=pos_prompt,
                     negative_prompt=neg_prompt,
                     num_inference_steps=num_inference_steps,
                     guidance_scale=12,).images[0]
        image.save(output_path)
        print(f"Image saved to {output_path}")

    elif mode=="background":
        image = pipe(prompt=pos_prompt,
                     negative_prompt=neg_prompt,
                     height=720,
                     width=1080,
                     num_inference_steps=num_inference_steps,
                     guidance_scale=12).images[0]
        image.save(output_path)
        print(f"Background template image saved to {output_path}")
    else:
        raise ValueError(f"Invalid mode '{mode}'. Supported modes are 'general' and 'background'.")

    del pipe
    cleanup()

    return output_path


def generate_image_api(api_key, prompt, output_path="image.png"):
    """
    Generates an image using OpenAI's DALL-E 2 API.
    Args:
        api_key (str): Your OpenAI API key.
        prompt (str): The prompt for the image generation.
        output_path (str): Path to save the generated image.
    Returns:
        str: Path to the saved image.
    """
    if not api_key:
        raise ValueError("API key is required for OpenAI image generation.")

    client = openai.OpenAI(api_key=api_key)

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt+", in the style of vector images",
        size="1024x1024",
        quality="standard",
        n=1
    )
    image_url = response.data[0].url
    img_data = requests.get(image_url).content
    with open(output_path, "wb") as f:
        f.write(img_data)
    print(f"Image saved to {output_path}")
    return output_path
