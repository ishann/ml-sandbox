import re
import json

def postprocess_ollama_output(raw_output):
    """
    Extracts structured slide data from the raw output of an LLM run via Ollama.
    Uses regex to find a ```json ... ``` block in the LLM output.
    Cleans up common hallucinations and formatting issues.
    Parses the JSON and returns structured slide data.
    
    Args:
        raw_output (str): The raw output string from the LLM.
    Returns:
        list: A list of dictionaries representing slides, each with a title and content.
    Raises:
        ValueError: If the output does not contain a valid JSON block.
    """
    match = re.search(r"```json\s*(.*?)```", raw_output, re.DOTALL)
    if not match:
        raise ValueError("No valid ```json ... ``` block found in the output.")
    json_text = match.group(1).strip()

    # Clean up common hallucinations
    json_text = json_text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    json_text = json_text.replace("–", "-").replace("—", "-")
    json_text = re.sub(r'"\s*(")', r'",\n\g<1>', json_text)
    json_text = re.sub(r'\n\s*-\s*([^\n"]+)', r'\n  "- \1"', json_text)
    json_text = re.sub(r',\s*(\]|\})', r'\1', json_text)

    # Truncate incomplete JSON
    last_bracket = max(json_text.rfind("]"), json_text.rfind("}"))
    if last_bracket != -1:
        json_text = json_text[:last_bracket+1]

    return json.loads(json_text)

