from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

def ingest(pdf_path, token_thresh=1000, text_thresh=100):
    """
    Imports a single PDF file.
    Splits PDF into Document chunks.
    Counts number of tokens, and shortens text before returning it.
    """

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)
    
    # Does not assume the cl100k_base encoding has been downloaded.
    # Downloads it every time.
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    token_count = 0
    for text in texts:
        token_count += num_tokens_from_string(text.page_content, encoding)
    
    if token_count>token_thresh:
        print("Too many tokens in raw text. Shortening...")
        texts = texts[:text_thresh]
    
    return texts


def num_tokens_from_string(string, encoding):
    """
    Returns the number of tokens in a text string.
    """
    num_tokens = len(encoding.encode(string))

    return num_tokens

