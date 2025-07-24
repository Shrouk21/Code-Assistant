from dotenv import load_dotenv
import os
from datasets import load_dataset
import re
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_together import Together
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


load_dotenv()

os.environ['TOGETHER_API_KEY'] = '2b9478ae0d7a9ab78f23c0185bd6723f190e3db0662119cc0d7b01bb5733ed30'

llm = Together(
    model='deepseek-ai/deepseek-coder-6.7b-instruct',
    temperature=0.2,
    max_tokens=512
)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

# ------------------Prepare Vector Store (Load if exists, otherwise create) -----------------
persist_directory = './chroma_code_db'

if not os.path.exists(persist_directory):
    print("Persistent directory not found. Creating and populating vector store...")
    ds = load_dataset("openai_humaneval")["test"]
    examples = [{"id": row["task_id"], "prompt": row["prompt"], "solution": row["canonical_solution"]} for row in ds]
    documents = [
        Document(
            page_content=f"{ex['prompt']}\n\n# Solution:\n{ex['solution']}",
            metadata={"id": ex["id"]}
        )
        for ex in examples
    ]

    def split_code_by_function(doc: Document) -> list[Document]:
        text = doc.page_content
        matches = list(re.finditer(r"^def\s+\w+\(.*?\):", text, re.MULTILINE))
        if not matches:
            return [doc]
        
        docs = []
        starts = [m.start() for m in matches] + [len(text)]
        
        for i in range(len(starts) - 1):
            chunk = text[starts[i]:starts[i+1]].strip()
            if chunk:
                docs.append(Document(page_content=chunk, metadata=doc.metadata))
        
        return docs

    split_documents = []
    for doc in documents:
        split_documents.extend(split_code_by_function(doc))

    vectorstore = Chroma.from_documents(
        documents=split_documents,
        embedding=embeddings,
        persist_directory=persist_directory  
    )
    print("Vector store created and persisted.")
else:
    print("Loading existing vector store from persistent directory...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Vector store loaded.")

retriever_tool = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})


#--------------------tools -------------------------------------
#Retrievel Agent
@tool
def retriever(query: str) -> str:
    """
    This tool used for generate_code node to search the information in the rag and 
    return similar codes
    """
    docs = retriever_tool.invoke(query)
    if not docs:
        return "I found no relevant code examples to the task you asked for"
    

    results = []

    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)



tools = [retriever]

# ------------------AI Agent------------------------------------

class StateAgent(TypedDict):
    message:Annotated[Sequence[BaseMessage], add_messages]
    task: str #generate or explain



# def chat(state: StateAgent) -> StateAgent:
#     pass




def chat(state: StateAgent) -> StateAgent:
    user_input = state['message'][-1].content
    prompt = f"""
You are a code assistant. Classify the user's request as one of the following tasks:
- "generate" → if the user wants code to be written.
- "explain" → if the user wants code to be explained.

Only respond with: generate or explain.

User input:
{user_input}
    """

    result=llm.invoke(prompt).strip().lower()
    task = result if result in {'generate', 'explain'} else 'fallback'
    return {
        **state,
        'task': task
    }

def router(state: StateAgent) -> str:
    return state['task']

def generate_code(state: StateAgent) -> StateAgent:
    user_input = state['message'][-1].content
    context = retriever(user_input)
    prompt = f"""You are an expert code generator.

            Below are relevant code snippets from previous solutions:
            {context}

            Now generate a complete solution for the following request:
            {user_input}
            """

    #Generate response using the LLM
    output = llm.invoke(prompt)

    return {
        **state,
        "message": state["message"] + [HumanMessage(content=prompt), SystemMessage(content=output)]
    }


def explain_code(state: StateAgent) -> StateAgent:
    user_input = state['message'][-1].content
    prompt = f"""You are an expert programmer and technical writer.

    Your task is to explain the following Python code in simple terms so that a junior developer or student can understand it.

    Explain what each part does and the overall purpose of the code. Be clear and concise.

    CODE:
    {user_input}
    """
    output= llm.invoke(prompt)
    return {
        **state,
        'message': state['message'] + [HumanMessage(content=prompt), SystemMessage(content=output)]
    
    }

def fallback(state: StateAgent) -> StateAgent:
    """A fallback node for when the router cannot determine the user's intent."""
    user_input = state['message'][-1].content
    output = f"I'm sorry, I couldn't determine if you wanted to 'generate' or 'explain' from your request: '{user_input}'. Please clarify."
    return {
        **state,
        'message': state['message'] + [SystemMessage(content=output)]
    }


graph = StateGraph(StateAgent)
graph.add_node('chat', chat)
graph.add_node('generate_code', generate_code)
graph.add_node('explain_code', explain_code)
graph.add_node('fallback', fallback)

graph.set_entry_point('chat')

graph.add_conditional_edges(
    'chat',
    router,
    {
        'generate': 'generate_code',
        'explain': 'explain_code',
        'fallback': 'fallback'
    } 
)

graph.add_edge('generate_code', END)
graph.add_edge('explain_code', END)
graph.add_edge('fallback', END)

app = graph.compile()

# ------ Visualize-----------------
# The `display` function is for interactive environments like Jupyter notebooks.
# To see the graph from a script, it's better to save it to a file.
# try:
#     with open("Smart Code Assitant/smart_assistant.png", "wb") as f:
#         f.write(app.get_graph().draw_mermaid_png())
#     print("\nGraph visualization saved to conditional_graph.png")
# except Exception as e:
#     print(f"\nCould not save graph image. You may need to install playwright (`pip install playwright` and `playwright install`). Error: {e}")


# ------main---------
def main():
    print("\n=== CODE ASSISTANT =====")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        result = app.invoke({"message": messages})
        
        print("\n=== ANSWER ===")
        print(result['message'][-1].content)

if __name__ == "__main__":
    main()