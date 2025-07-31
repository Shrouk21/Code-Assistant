import os
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "./chroma_code_db"
EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
OLLAMA_MODEL_NAME = "codellama:7b"
