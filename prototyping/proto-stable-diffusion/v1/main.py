import argparse
import os

from dotenv import load_dotenv
from exporter.ppt_exporter import build_ppt
from generator.visual_content_builder import generate_images
from models.summarizer import summarize_text_local_llm
from utils.pdf_reader import extract_text_from_pdf
from time import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main(data_path, meta_path, title, output_path):

    init = time()

    # Extract raw text. No LLMs.
    raw_text = extract_text_from_pdf(data_path)
    meta_text = extract_text_from_pdf(meta_path)

    # Summarize text using LLMs.
    slide_content = summarize_text_local_llm(raw_text)

    # Generate images. Return templates and slides.
    templates, slides = generate_images(
        slide_content, meta_text, api_key=OPENAI_API_KEY, genimg_lvl=1)

    # Build and export a PPTX.
    build_ppt(templates, slides, title, output_path)

    print(f"Presentation saved to {output_path}.")
    fin = time()
    print(f"Elapsed: {fin-init:.2f}s.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True,
                        help='Input PDF file.')
    parser.add_argument('--meta', required=True,
                        help='Mission statement/ charter.')
    parser.add_argument('--genimg_lvl', type=int, default=1, choices=[0, 1, 2],
                        help='Image generation level: 0 = Placeholder, 1 = Local SD model, 2 = Dall-E-2 through OpenAI API.')
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.data))[0]
    output_path = os.path.join(f"./data/outputs/{base_name}.pptx")
    title = ["Attention is All You Need", "A Presentation on Transformers"]

    main(args.data, args.meta, title, output_path)
