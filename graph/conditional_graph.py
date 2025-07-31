from langgraph.graph import StateGraph, END
from agents.state import StateAgent
from agents.nodes import chat, generate_code, explain_code, fallback, router

def get_app():
    graph = StateGraph(StateAgent)
    graph.add_node('chat', chat)
    graph.add_node('generate_code', generate_code)
    graph.add_node('explain_code', explain_code)
    graph.add_node('fallback', fallback)
    
    graph.set_entry_point('chat')
    graph.add_conditional_edges('chat', router, {
        'generate': 'generate_code',
        'explain': 'explain_code',
        'fallback': 'fallback'
    })
    graph.add_edge('generate_code', END)
    graph.add_edge('explain_code', END)
    graph.add_edge('fallback', END)
    
    return graph.compile()
