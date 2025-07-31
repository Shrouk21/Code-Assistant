#-------------------------------Classification Prompt-----------------------------
def classify_prompt(user_input: str) -> str:
    return f"""You are an expert AI assistant. Decide the user's intent: do they want code to be *generated* or *explained*?

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

Expected output: just one word (e.g. generate or explain or something else).
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
def fallback_prompt(user_input: str) -> str:
    return f"""You are an expert programmer.

A user gave the following input:
\"\"\"{user_input}\"\"\"

Your task is:
1. Decide if the request is related to code (e.g., asking to generate or explain code).
2. If the input is AMBIGUOUS or unclear, respond with: 
   "I don't fully understand your question. Could you please clarify what you want me to do?"
3. If the input is IRRELEVANT to code, respond with:
   "Your question seems unrelated to programming, but here's my best answer:" — and then proceed to answer.

Be honest, concise, and helpful.
"""
