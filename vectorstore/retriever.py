# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from config.settings import EMBEDDING_MODEL_NAME, PERSIST_DIR

# def get_embedding_model():
#     return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# def get_vectorstore():
#     return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding_model())

# def get_retriever():
#     return get_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": 5})
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config.settings import EMBEDDING_MODEL_NAME, PERSIST_DIR

# Create a singleton embedding model to ensure consistency
_embedding_model = None

def get_embedding_model():
    """Get a singleton embedding model instance"""
    global _embedding_model
    if _embedding_model is None:
        print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            #model_kwargs={'device': 'gpu'},  # Explicitly set device
            encode_kwargs={'normalize_embeddings': True}  # Ensure consistent normalization
        )
    return _embedding_model

def get_vectorstore():
    """Get the Chroma vectorstore"""
    embedding_model = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR, 
        embedding_function=embedding_model
    )
    return vectorstore

def get_retriever(k=5):
    """Get the retriever with specified number of results"""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": k}
    )