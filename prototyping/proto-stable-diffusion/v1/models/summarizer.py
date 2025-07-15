import subprocess
import json
import re

from models.prompts import PROMPT
from utils.memory_management import cleanup
from utils.postprocess_llm_output import postprocess_ollama_output

def summarize_text_local_llm(text, model="mixtral", timeout=120):
    """
    Summarizes the input text using a local LLM model via Ollama.
    Args:
        text (str): The input text to summarize.
        model (str): The name of the local LLM model to use (default: "mixtral").
        timeout (int): Timeout for the subprocess call in seconds (default: 120).
    Returns:
        list: A list of dictionaries representing slides, each with a title and content.

    Takeaways:
    1. A locally running ollama instance loads in either mistral or mixtral
       and generates structured output for the PPTX.
    2. It does require post-processing because locally run LLMs can only go so far.
    3. The diversity of outputs is extremely high...!        
    """

    if not text:
        return [{
            "type": "content",
            "title": "No Content",
            "content": ["The input text is empty."]
        }]

    if len(text) > 10000:
        text = text[:10000]
        print("Input text truncated to 10,000 characters for processing.")

    # Prepare the prompt for the LLM
    prompt = PROMPT.format(input_chunk=text) 

    try:
        print(f"Running {model} locally to summarize text.")
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            capture_output=True,
            timeout=timeout
        )
        output = result.stdout.decode("utf-8")
        return postprocess_ollama_output(output)

    except Exception as e:
        print("Summarization failed:", e)
        return [{
            "type": "content",
            "title": "Error Generating Slides",
            "content": [str(e)]
        }]

    finally:
        print("Summarization complete. Cleaning up memory.")
        cleanup()


def summarize_text_nlp(text, max_sentences_per_slide=3):
    """
    Summarizes the input text using classic NLP techniques.
    Args:
        text (str): The input text to summarize.
        max_sentences_per_slide (int): Maximum number of sentences per slide (default: 3).
    Returns:
        list: A list of dictionaries representing slides, each with a title and bullets.
        
    Takeaways:
        1. Non-LLM. Classic NLP. Uses spaCy for sentence segmentation.
        2. Generates slides with a title and bullet points.
        3. Surprisingly robust at being useless.
    """

    print("\n\nWARNING!!\nThis code is not being used in any versions of the prototype and might break.\n\n")

    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    slides = []
    current_slide = {"title": "", "bullets": []}

    sentence_buffer = []
    slide_index = 1

    for sent in doc.sents:
        if not current_slide["title"]:
            current_slide["title"] = sent.text.strip()[:60]

        sentence_buffer.append(sent.text.strip())

        if len(sentence_buffer) >= max_sentences_per_slide:
            current_slide["bullets"] = sentence_buffer.copy()
            slides.append(current_slide)
            sentence_buffer.clear()
            slide_index += 1
            current_slide = {"title": "", "bullets": []}

    if sentence_buffer:
        if not current_slide["title"] and sentence_buffer:
            current_slide["title"] = sentence_buffer[0][:60]
        current_slide["bullets"] = sentence_buffer
        slides.append(current_slide)

    return slides
