from langchain_core.tools import tool
from vectorstore.retriever import get_retriever

retriever_tool = get_retriever()

@tool
def retriever(query: str) -> str:
    """
    Searches the code vectorstore for examples similar to the query.
    """
    docs = retriever_tool.invoke(query)
    if not docs:
        return "No relevant code examples found."

    return "\n\n".join(f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))

tools = [retriever]
