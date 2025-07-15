"""
Compared to summarizer, chunky_summarizer attempts to chunk the data before sending it through the locally
running Mixtral7-8B, but this is probably not required if we switch to OpenAI API-based text summarizers.
"""
import json
import re
from typing import List
import subprocess

def chunk_text(text: str, max_words: int = 600) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk.strip())
    return chunks

def extract_slides_from_ollama_output(raw_output: str) -> List[dict]:
    match = re.search(r"```json\s*(.*?)```", raw_output, re.DOTALL)
    if match:
        json_text = match.group(1).strip()
    else:
        json_match = re.search(r'(\[.*?\])', raw_output, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON-like structure found in the output.")
        json_text = json_match.group(1).strip()

    # Fix curly quotes and common formatting problems
    json_text = json_text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    json_text = json_text.replace("–", "-").replace("—", "-")
    json_text = re.sub(r'"\s*(")', r'",\n\g<1>', json_text)
    json_text = re.sub(r'\n\s*-\s*([^\n"]+)', r'\n  "- \1"', json_text)
    json_text = re.sub(r',\s*(\]|\})', r'\1', json_text)
    last_bracket = max(json_text.rfind("]"), json_text.rfind("}"))
    if last_bracket != -1:
        json_text = json_text[:last_bracket+1]

    return json.loads(json_text)

def summarize_text_llm_chunked(input_text, model_name="mixtral"):
    # Add a docstring for this function.
    """
    Summarizes the input text using a local LLM model via Ollama, processing it
    in chunks. The difference from summarize_text_llm is that this function
    chunks the input text into smaller parts before sending them to the LLM,
    which can help with large inputs.

    Args:
        input_text (str): The input text to summarize.
        model_name (str): The name of the local LLM model to use (default: "mixtral").
        timeout (int): Timeout for the subprocess call in seconds (default: 120).
    Returns:
        list: A list of dictionaries representing slides, each with a title and content.
    """

    from models.prompts import PROMPT  # Or inline it if needed

    chunks = chunk_text(input_text)
    all_slides = []

    for i, chunk in enumerate(chunks):
        print(f"\n🔹 Processing chunk {i+1}/{len(chunks)}...")

        prompt = PROMPT.format(input_chunk=chunk)
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )

        output = result.stdout.decode("utf-8")
        try:
            slides = extract_slides_from_ollama_output(output)
            all_slides.extend(slides)
        except Exception as e:
            print(f"Failed to parse chunk {i+1}: {e}")

    return all_slides
