from dotenv import load_dotenv
import os
from datasets import load_dataset
import re
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


load_dotenv()

# Initialize Ollama LLM (Free local model)
llm = Ollama(
    model='codellama:7b',  # You can also use 'deepseek-coder:6.7b' or 'llama2:7b'
    temperature=0.2
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
    classification: str  # Store the raw classification result



def chat(state: StateAgent) -> StateAgent:
    user_input = state['message'][-1].content
    prompt = f"""
    You are an expert AI assistant. Decide the user's intent: do they want code to be *generated* or *explained*?

    Classify only as:
    - generate
    - explain

    Examples:
    - "write a function to sort a list" → generate
    - "explain this function: def foo(x): return x+1" → explain
    - "generate function that add two numbers" → generate
    - "give me function that..." → generate

    User input:
    {user_input}
    """


    # result = llm.invoke(prompt).strip().lower()
    # task = result['message']['content'].strip().lower()
    # classification = result  # Store the raw result for display
    # task = result if result in {'generate', 'explain'} else 'fallback'
    result = llm.invoke(prompt)
    classification = result.strip()

    # Extract the quoted keyword from the sentence
    match = re.search(r'"(generate|explain)"', classification.lower())
    task = match.group(1) if match else 'fallback'

    return {
        **state,
        'task': task,
        'classification': classification
    }

def router(state: StateAgent) -> str:
    return state['task']

def generate_code(state: StateAgent) -> StateAgent:
    user_input = state['message'][-1].content
    context = retriever(user_input)
    prompt = f"""You are an expert code generator.

            Below are relevant code snippets from previous solutions:
            {context}

            Now generate a complete Python function for the following request:
            {user_input}

            Requirements:
            - Provide ONLY the function code
            - Include proper function definition with parameters
            - Add docstring if appropriate
            - Make it ready to use
            """

    #Generate response using the LLM
    output = llm.invoke(prompt)

    return {
        **state,
        "message": state["message"] + [HumanMessage(content=prompt), SystemMessage(content=output)]
    }


def explain_code(state: StateAgent) -> StateAgent:
    user_input = state['message'][-1].content
    
    # Check if the input contains actual code
    if not any(keyword in user_input.lower() for keyword in ['def ', 'class ', 'import ', 'for ', 'if ', 'while ', '=', 'print', 'return']):
        # If it doesn't look like code, ask for clarification
        output = f"I don't see any code in your input: '{user_input}'. Please provide the Python code you'd like me to explain."
    else:
        prompt = f"""You are an expert programmer and technical writer.

        Your task is to explain the following Python code in simple terms so that a junior developer or student can understand it.

        Explain:
        1. What each part does
        2. The overall purpose of the code
        3. How it works step by step

        Be clear and concise.

        CODE:
        {user_input}
        """
        output = llm.invoke(prompt)
    
    return {
        **state,
        'message': state['message'] + [HumanMessage(content=prompt if 'prompt' in locals() else ""), SystemMessage(content=output)]
    
    }

def fallback(state: StateAgent) -> StateAgent:
    """A fallback node for when the router cannot determine the user's intent."""
    user_input = state['message'][-1].content
    output = f"""I couldn't determine if you wanted to 'generate' or 'explain' from your request: '{user_input}'.

Please clarify:
- If you want me to write/create code, say something like: "Write a function that..."
- If you want me to explain code, provide the code and say: "Explain this code:"

Examples:
- "Write a function to reverse a string"
- "Explain this code: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
"""
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
    print("\n=== CODE ASSISTANT WITH OLLAMA =====")
    print("💡 Examples:")
    print("  - Generate: 'Write a function to find prime numbers'")
    print("  - Explain: 'Explain this code: def hello(): print(\"hi\")'")
    print("  - Type 'exit' or 'quit' to stop")
    
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type

        try:
            result = app.invoke({"message": messages})
            
            # Display classification
            classification = result.get('classification', 'unknown')
            print(f"\n🔍 Task Classification: {classification.upper()}")
            
            print("\n=== ANSWER ===")
            print(result['message'][-1].content)
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure Ollama is running and the model is installed.")
            print("Run: ollama serve (in another terminal)")
            print("Run: ollama pull codellama:7b")

if __name__ == "__main__":
    main()