"""
Embed and init vector store.
"""
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def embed(texts, openai_api_key):
    """
    Embed texts and initialize the vector store.
    """
    # Compute embeddings.
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = FAISS.from_documents(documents=texts, embedding=embedding)

    return vectordb