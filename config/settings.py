import os
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "./chroma_code_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"
OLLAMA_MODEL_NAME = "codellama:7b"
