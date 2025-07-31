#-------------------------------Classification Prompt-----------------------------
# def classify_prompt(user_input: str) -> str:
#     return f"""You are an expert AI assistant. Decide the user's intent: do they want code to be *generated* or *explained*?

# Classify only as:
# - generate
# - explain

# Examples:
# - "write a function to sort a list" → generate
# - "explain this function: def foo(x): return x+1" → explain
# - "generate function that add two numbers" → generate
# - "give me function that..." → generate

# User input:
# {user_input}

# Expected output: just one word (e.g. generate or explain or something else).
# """
def classify_prompt(user_input: str) -> str:
    return f"""You are an expert coding assistant.

Your task is to classify the user's intent based on their input. Choose one of the following categories:

- generate → if they are asking you to write or create code.
- explain → if they are asking you to analyze, interpret, or describe existing code.
- unclear → if the input is ambiguous or irrelevant to code generation or explanation.

Examples:
- "write a function to sort a list" → generate
- "explain this function: def foo(x): return x+1" → explain
- "generate function that adds two numbers" → generate
- "give me a function that reverses a string" → generate
- "what is the time complexity of this code?" → explain
- "I love pizza!" → unclear
- "what's the best programming language?" → unclear
- "can you help me?" → unclear

User input:
{user_input}

Respond with only one word: generate, explain, or unclear.
"""

#------------------------------------Explanation Prompt----------------------------------------------
def explain_prompt(code: str) -> str:
    return f"""You are an expert programmer and technical writer.

Your task is to explain the following Python code in simple terms so that a junior developer or student can understand it.

Explain:
1. What each part does
2. The overall purpose of the code
3. How it works step by step

Be clear and concise.

CODE:
{code}
"""

#------------------------------------Generate Prompt----------------------------------------------
def generate_prompt(user_input: str, context: str) -> str:
    return f"""You are an expert code generator.

Below are relevant code snippets from previous solutions:
{context}

Now generate a complete Python function for the following request:
{user_input}

Requirements:
- Provide ONLY the function code
- Include proper function definition with parameters
- Add a docstring if appropriate
- Make it ready to use
"""

#------------------------------------Fallback Prompt----------------------------------------------
# def fallback_prompt(user_input: str) -> str:
#     return f"""You are an expert programmer.

# A user gave the following input:
# \"\"\"{user_input}\"\"\"

# Your task is:
# 1. Decide if the request is related to code (e.g., asking to generate or explain code).
# 2. If the input is AMBIGUOUS or unclear, respond with: 
#    "I don't fully understand your question. Could you please clarify what you want me to do?"
# 3. If the input is IRRELEVANT to code, respond with:
#    "Your question seems unrelated to programming, but here's my best answer:" — and then proceed to answer.

# Be honest, concise, and helpful.
# """
def fallback_prompt(user_input: str) -> str:
    return f"""You are a concise and helpful AI programming assistant.

Here is the user's message:
\"\"\"{user_input}\"\"\"

Decide which of the following applies **and respond directly without explanation**:

1. If the input is related to code (e.g., a request to generate or explain code), do nothing — leave the response blank.
2. If the input is **ambiguous** (unclear what the user wants), reply with:
   I don't fully understand your question. Could you please clarify what you want me to do?
3. If the input is **irrelevant** (not related to programming), reply with:
   Your question seems unrelated to programming, but here's my best answer: — and then provide your best helpful answer.

Do not explain your decision. Just respond with the appropriate message only.
"""


