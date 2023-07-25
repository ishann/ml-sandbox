import os
from dotenv import load_dotenv, find_dotenv

def init():
    """
    Get things up and running.
    """
    load_dotenv(find_dotenv())
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    return OPENAI_API_KEY


