from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config.settings import EMBEDDING_MODEL_NAME, PERSIST_DIR

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_vectorstore():
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding_model())

def get_retriever():
    return get_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": 5})
