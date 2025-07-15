from generator.image_generator import generate_image_local, generate_image_api
from models.meta_prompter import generate_image_prompt, generate_template_prompts
from utils.images import make_translucent
import ipdb

def generate_images(slide_data, meta_data, api_key=None, genimg_lvl=0):
    """
    Build slides from the summarized data, generating images if required.
    Args:
        api_key (str): OpenAI API key for image generation.
        slide_data (list): List of dictionaries containing slide content.
        genimg_lvl (int): Level of image generation to use.
        genimg_lvl: 0 = Placeholder
                    1 = Local SD model (terrible quality).
                    2 = Dall-E-2 through OpenAI API (good quality but $$).
    Returns:
        list: List of slides with images added.
    """
    if not slide_data:
        raise ValueError("No summary data provided for slide generation.")
    if genimg_lvl not in [0, 1, 2]:
        raise ValueError("Invalid image generation level. Use 0, 1, or 2.")
    if genimg_lvl == 2 and not api_key:
        raise ValueError("API key is required for image generation level 2.")

    # Conditioned on the meta_data, generate the following templates
    # conditioned on the metadata:
    # 1. Title slide.
    # 2. General a common slide templates to be used for all other slides:
    #    section, content, visual.
    templates = {"title_slide": None, "slides": None}

    intro, general = generate_template_prompts(meta_data, model="mixtral")
    templates["title_slide"] = generate_image_local(intro["positive_prompt"],
                                                    intro["negative_prompt"],
                                                    mode="background",
                                                    output_path=f"./data/tmp/title_bg.png",
                                                    num_inference_steps=4)
    templates["slides"] = generate_image_local(general["positive_prompt"],
                                               general["negative_prompt"],
                                               mode="background",
                                               output_path=f"./data/tmp/slide_bg.png",
                                               num_inference_steps=4)

    make_translucent(templates["slides"], templates["slides"], alpha_factor=0.75)

    slides = []
    for idx, item in enumerate(slide_data):

        if "image_prompt" in item:
            pos_prompt, neg_prompt = generate_image_prompt(
                f"{item["title"]}: {item["image_prompt"]}"
            )
            print(f"Generating an image at {idx+1}")
        else:
            # This slide does not require image generation.
            slides.append(item)
            continue        

        if genimg_lvl==0:
            # Use placeholder.
            image_path = "./data/tmp/dalle3.png"
        elif genimg_lvl==1:
            # Use local SDXL through HF's diffusers.
            image_path = generate_image_local(pos_prompt,
                                              neg_prompt,
                                              mode="general",
                                              output_path=f"./data/tmp/{item['title'].replace(' ', '_')}.png",
                                              num_inference_steps=4)
        elif genimg_lvl==2:
            # Use OpenAI API calls to DallE-3.
            image_path = generate_image_api(api_key, pos_prompt, output_path=f"./data/tmp/{item['title'].replace(' ', '_')}.png")
        else:
            raise ValueError("Incorrect mode for image generation.")
        item["image"] = image_path
        slides.append(item)

    return templates, slides
