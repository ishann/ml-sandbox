import json
import subprocess
import sys
import warnings
import ipdb

from models.prompts import META_IMG_PROMPT, META_BG_PROMPT
from models.summarizer import postprocess_ollama_output
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def warn_to_stdout(message):
    warnings.showwarning = lambda msg, *args, **kwargs: print(f"WARNING: {msg}", file=sys.stdout)
    warnings.warn(message)


def generate_image_prompt(concept, model="mixtral"):
    # Update the docstring to reflect that we now generate both positive and negative prompts.

    """
    Generate concise visual prompts (both positive and negative) for
    a technical concept using an LLM.
    LLMs (atleast small local ones) can be bad at hierarchical tasks.
    Hence, the optional image_prompt should not be expected to be detailed
    (and also not tailored to work directly for another small Stable Diff model).
    Solution: Each terse image_prompt is fed into another Mixtral instance with a prompt.

    Args:
        concept (str): The technical concept to visualize.
        model (str): The LLM model to use for generating the prompt.
    Returns:
        str: The generated positive image prompt.
        str: The generated negative image prompt.
    Raises:
        ValueError: If the concept is empty or the generated prompt is invalid.
    
    NOTE: CLIP can only take 89 words in its text prompts. Hence, the magic number, ie, 88.
    """
    if not concept:
        raise ValueError("Concept must be a non-empty string.")

    print(f"{model} is writing prompts for Stable Diffusion XL.")
    meta_prompt = META_IMG_PROMPT.format(concept=concept)

    result = subprocess.run(
        ["ollama", "run", model],
        input=meta_prompt.encode(),
        capture_output=True,
        timeout=120
    )

    result = result.stdout.decode("utf-8")
    # Add try except block to handle JSON parsing errors.
    try :
        output = json.loads(result)
    except:
        output = postprocess_ollama_output(result)

    if not output or "positive_prompt" not in output or "negative_prompt" not in output:
        raise ValueError("Generated prompts are empty or invalid.")

    positive_prompt = output["positive_prompt"]
    negative_prompt = output["negative_prompt"]

    pos_tokens = check_prompt_length(positive_prompt)
    neg_tokens = check_prompt_length(negative_prompt)

    if not positive_prompt or not negative_prompt:
        raise ValueError("Generated prompt is empty or invalid.")
    if pos_tokens > 75:
        warn_to_stdout("positive prompt exceeds 75 tokens, which is beyond clip's limits, and will be truncated by SDXL.")
    if neg_tokens > 75:
        warn_to_stdout("negative prompt exceeds 75 tokens, which is beyond clip's limits, and will be truncated by SDXL.")
    
    return positive_prompt, negative_prompt


def generate_template_prompts(meta_text, model="mixtral"):
    """
    Generate background image prompts (positive + negative) for intro and general slides
    based on a mission statement or team charter using a local Mixtral LLM.
    """

    if not meta_text:
        raise ValueError("Meta text must be a non-empty string.")

    print(f"{model} is writing template generation prompts for SDXL.")
    meta_prompt = META_BG_PROMPT.format(meta_text=meta_text)

    result = subprocess.run(
        ["ollama", "run", model],
        input=meta_prompt.encode(),
        capture_output=True,
        timeout=120
    )
    
    result = result.stdout.decode("utf-8")
    # Add try except block to handle JSON parsing errors.
    try :
        output = json.loads(result)
    except:
        output = postprocess_ollama_output(result)

    if not output:
        raise ValueError("Generated prompts are empty or invalid.")    

    intro = output["intro"]
    general = output["general"]

    # Optional token length checks if you're feeding into CLIP
    for label, p in [("intro positive", intro["positive_prompt"]),
                     ("intro negative", intro["negative_prompt"]),
                     ("general positive", general["positive_prompt"]),
                     ("general negative", general["negative_prompt"])]:
        tokens = check_prompt_length(p)
        if tokens > 75:
            warn_to_stdout(f"{label} prompt exceeds 75 tokens and may be truncated by SDXL.")

    return intro, general


def check_prompt_length(prompt, token_limit=75):
    # Add a docstring to clarify the function's purpose.
    """
    Check the length of a text prompt for CLIP compatibility.
    Args:
        prompt (str): The text prompt to check.
    Returns:
        int: The number of tokens in the prompt.
    Raises:
        ValueError: If the prompt is empty or exceeds CLIP's token limit.
    
    CLIP can handle up to 77 tokens; we use 75 to allow for
    start and end tokens to be appended.
    """
    if not prompt:
        raise ValueError("Prompt must be a non-empty string.")

    # Tokenize and count
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    num_tokens = len(tokens)

    if num_tokens > token_limit:
        print("Warning: exceeds CLIP limit of 77 tokens and will be truncated.")

    return num_tokens
