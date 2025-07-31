import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_chroma import Chroma
from vectorstore.retriever import get_embedding_model
from utils.code_splitter import split_code_by_function
from config.settings import PERSIST_DIR

def build_vectorstore():
    if os.path.exists(PERSIST_DIR):
        print("Loading existing Chroma DB...")
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embedding_model())

    print("Building new Chroma DB from dataset...")
    ds = load_dataset("openai_humaneval")["test"]
    examples = [{"id": row["task_id"], "prompt": row["prompt"], "solution": row["canonical_solution"]} for row in ds]

    raw_documents = [
        Document(
            page_content=f"{ex['prompt']}\n\n# Solution:\n{ex['solution']}",
            metadata={"id": ex["id"]}
        ) for ex in examples
    ]

    split_documents = [d for doc in raw_documents for d in split_code_by_function(doc)]

    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=get_embedding_model(),
        persist_directory=PERSIST_DIR
    )

    print("Chroma DB created and persisted.")
    return vectorstore
