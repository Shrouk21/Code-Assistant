# import os
# from datasets import load_dataset
# from langchain_core.documents import Document
# from langchain_chroma import Chroma
# from vectorstore.retriever import get_embedding_model
# from utils.code_splitter import split_code_by_function
# from config.settings import PERSIST_DIR

# def build_vectorstore():
#     if os.path.exists(PERSIST_DIR):
#         print("Loading existing Chroma DB...")
#         return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding_model())

#     print("Building new Chroma DB from dataset...")
#     ds = load_dataset("openai_humaneval")["test"]
#     examples = [{"id": row["task_id"], "prompt": row["prompt"], "solution": row["canonical_solution"]} for row in ds]

#     raw_documents = [
#         Document(
#             page_content=f"{ex['prompt']}\n\n# Solution:\n{ex['solution']}",
#             metadata={"id": ex["id"]}
#         ) for ex in examples
#     ]

#     split_documents = [d for doc in raw_documents for d in split_code_by_function(doc)]

#     vectorstore = Chroma.from_documents(
#         documents=split_documents,
#         embedding=get_embedding_model(),
#         persist_directory=PERSIST_DIR
#     )

#     print("Chroma DB created and persisted.")
#     return vectorstore
import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_chroma import Chroma
from vectorstore.retriever import get_embedding_model
from utils.code_splitter import split_code_by_function
from config.settings import PERSIST_DIR

def build_vectorstore(force_rebuild=False):
    """Build or load the vectorstore"""
    
    # Check if vectorstore exists and is not empty
    if os.path.exists(PERSIST_DIR) and not force_rebuild:
        print("Loading existing Chroma DB...")
        try:
            vectorstore = Chroma(
                persist_directory=PERSIST_DIR, 
                embedding_function=get_embedding_model()
            )
            
            # Test if the vectorstore has documents
            test_results = vectorstore.similarity_search("def", k=1)
            if test_results:
                print(f"Loaded existing vectorstore with {len(test_results)} sample documents")
                return vectorstore
            else:
                print("Existing vectorstore appears empty, rebuilding...")
        except Exception as e:
            print(f"Error loading existing vectorstore: {e}")
            print("Rebuilding vectorstore...")
    
    print("Building new Chroma DB from dataset...")
    
    # Load dataset
    ds = load_dataset("openai_humaneval")["test"]
    examples = [
        {
            "id": row["task_id"], 
            "prompt": row["prompt"], 
            "solution": row["canonical_solution"]
        } 
        for row in ds
    ]
    
    print(f"Loaded {len(examples)} examples from dataset")
    
    # Create raw documents
    raw_documents = [
        Document(
            page_content=f"{ex['prompt']}\n\n# Solution:\n{ex['solution']}",
            metadata={"id": ex["id"], "type": "code_example"}
        ) 
        for ex in examples
    ]
    
    # Split documents by function
    split_documents = []
    for doc in raw_documents:
        split_docs = split_code_by_function(doc)
        split_documents.extend(split_docs)
    
    print(f"Split into {len(split_documents)} document chunks")
    
    # Remove existing directory if rebuilding
    if force_rebuild and os.path.exists(PERSIST_DIR):
        import shutil
        shutil.rmtree(PERSIST_DIR)
        os.makedirs(PERSIST_DIR, exist_ok=True)
    
    # Create vectorstore
    embedding_model = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR
    )
    
    print("Chroma DB created and persisted.")
    
    # Verify the vectorstore was created successfully
    test_results = vectorstore.similarity_search("def", k=1)
    print(f"Verification: Found {len(test_results)} test results")
    
    return vectorstore