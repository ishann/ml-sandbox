
from src.engine import create_presentation_from_pdf

# Config
DEVICE = "mps"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
TEMP_IMAGE_DIR = "./data/tmp/gen_imgs"
PDF_PATH = "./data/inputs/Transformers.pdf"
OUTPUT_PPTX = "./data/outputs/Transformers.pptx"


if __name__ == "__main__":
    create_presentation_from_pdf(PDF_PATH)