from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import StateAgent
from prompts.prompts import (
    classify_prompt,
    generate_prompt,
    explain_prompt,
    fallback_prompt
)
from tools.tools import retriever
from langchain_community.llms import Ollama
from config.settings import OLLAMA_MODEL_NAME
import re

llm = Ollama(
    model=OLLAMA_MODEL_NAME,  # We can also use 'deepseek-coder:6.7b' or 'llama2:7b'
    temperature=0.2
)
def chat(state: StateAgent) -> StateAgent:
    user_input = state['message'][-1].content
    prompt = classify_prompt(user_input)
    result = llm.invoke(prompt)
    raw = result.strip().lower()
    match = re.search(r"(generate|explain|unclear)", raw)
    task = match.group(1) if match else 'unclear'
    return {**state, 'task': task, 'classification': raw}

def router(state: StateAgent) -> str:
    return state['task']

def generate_code(state: StateAgent) -> StateAgent:
    user_input = state['message'][-1].content
    context = retriever(user_input)
    prompt = generate_prompt(user_input, context)
    output = llm.invoke(prompt)
    return {
        **state,
        "message": state["message"] + [HumanMessage(content=prompt), SystemMessage(content=output)]
    }

def explain_code(state: StateAgent) -> StateAgent:
    user_input = state['message'][-1].content
    if not any(k in user_input.lower() for k in ['def ', 'class ', 'import ', 'for ', 'if ', 'while ', '=', 'print', 'return']):
        output = f"I don't see any code in your input: '{user_input}'. Please provide the Python code you'd like me to explain."
        return {**state, "message": state["message"] + [SystemMessage(content=output)]}
    prompt = explain_prompt(user_input)
    output = llm.invoke(prompt)
    return {
        **state,
        "message": state["message"] + [HumanMessage(content=prompt), SystemMessage(content=output)]
    }

def fallback(state: StateAgent) -> StateAgent:
    user_input = state['message'][-1].content
    prompt = fallback_prompt(user_input)
    output = llm.invoke(prompt)
    return {
        **state,
        "message": state["message"] + [SystemMessage(content=output)]
    }
