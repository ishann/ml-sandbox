import fitz
from diffusers import StableDiffusionPipeline

# Load PDF and extract text per page
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page_texts = [page.get_text() for page in doc]
    doc.close()
    return page_texts

# Generate image from summary
def generate_image(prompt, out_path):
    # Load Stable Diffusion
    sd_pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID).to(DEVICE)
    
    image = sd_pipe(prompt).images[0]
    image.save(out_path)
    return out_path

