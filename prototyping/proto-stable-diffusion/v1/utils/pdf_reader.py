"""
Document ingestion will require serious effort for multiple modalities
like text/ tables/ graphics/ general images/ equations, etc.

TODOs:
1. Add support for images.
2. Add support for equations.
3. Maybe just use LLM document readers for this...! 

See the bottom of this doc for a comprehensive plan.
"""
import fitz

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

"""
(Next version that can treat document ingestion as a first-class citizen.)
Document Parsing + Understanding Pipeline (Local)

1. Parse PDF content
   - Text + bbox: use PyMuPDF
   - Images: use pdf2image or PyMuPDF
   - Structure: detect tables, figures, headers with LayoutParser (PubLayNet)

2. OCR & Specialized Extraction
   - OCR (if needed): Tesseract or TrOCR
   - Equations: Im2LaTeX for LaTeX reconstruction
   - Tables: optionally use detectron2 or LayoutParser TableBank model

3. Multimodal Understanding / LLM
   - Donut (VisionEncoderDecoder) for end-to-end parsing
   - LayoutLMv3 or DocFormer for layout-aware understanding
   - LLaVA for multimodal (text+image) QA
   - phi-3-mini or mistral (via Ollama) for reasoning / orchestration

4. Optional toolkits
   - unstructured (document chunking/tagging)
   - doctr or parseq (OCR models)

All components can run locally on M4 Mac with MPS backend.
"""
