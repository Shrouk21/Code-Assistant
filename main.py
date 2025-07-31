from langchain_core.messages import HumanMessage
from graph.conditional_graph import get_app

def main():
    app = get_app()
    print("=== SMART CODE ASSISTANT ===")
    while True:
        user_input = input("Q: ")
        if user_input.lower() in {'exit', 'quit'}:
            break
        try:
            result = app.invoke({"message": [HumanMessage(content=user_input)]})
            print(f"\n🔍 Classified Task: {result.get('classification', 'unknown').upper()}")
            print("\n🧠 ANSWER:\n", result['message'][-1].content)
        except Exception as e:
            print("⚠️ Error:", e)

if __name__ == "__main__":
    main()
