import gradio as gr
from langchain_core.messages import HumanMessage
from graph.conditional_graph import get_app

# Initialize the app once
app = get_app()

def process_question(username, question):
    """Process the user's question and return classification and answer"""
    if not question.strip():
        return "", "Please enter a question!"
    
    try:
        # Invoke the langchain app
        result = app.invoke({"message": [HumanMessage(content=question)]})
        
        # Extract classification and answer
        classification = result.get('classification', 'unknown').upper()
        answer = result['message'][-1].content
        
        return classification, answer
        
    except Exception as e:
        return "ERROR", f"⚠️ Error: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Smart Code Assistant", theme=gr.themes.Soft()) as demo:
        # Store username in state
        username_state = gr.State("")
        
        # Welcome section
        with gr.Row():
            gr.Markdown("# 🤖 Smart Code Assistant")
        
        # Username input (initially visible)
        with gr.Row() as username_row:
            with gr.Column():
                gr.Markdown("### Welcome! Please enter your name to get started:")
                username_input = gr.Textbox(
                    placeholder="Enter your name...",
                    label="Your Name",
                    interactive=True
                )
                start_btn = gr.Button("Start Assistant", variant="primary")
        
        # Main interface (initially hidden)
        with gr.Row(visible=False) as main_interface:
            with gr.Column():
                # Personalized greeting
                greeting = gr.Markdown("")
                
                # Question input
                question_input = gr.Textbox(
                    placeholder="Ask me anything about code...",
                    label="Your Question",
                    lines=3
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Ask Question", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
                
                # Results section
                with gr.Row():
                    with gr.Column(scale=1):
                        classification_output = gr.Textbox(
                            label="🔍 Task Classification",
                            interactive=False,
                            lines=1
                        )
                    
                with gr.Row():
                    answer_output = gr.Textbox(
                        label="🧠 Answer",
                        interactive=False,
                        lines=10
                    )
                
                # Reset button
                with gr.Row():
                    reset_btn = gr.Button("Start Over", variant="secondary", size="sm")
        
        def start_session(name):
            """Initialize the session with username"""
            if not name.strip():
                gr.Warning("Please enter your name!")
                return gr.update(), gr.update(), gr.update(), ""
            
            greeting_text = f"## Hi, {name}! 👋 How can I help you today?"
            
            return (
                gr.update(visible=False),  # Hide username section
                gr.update(visible=True),   # Show main interface
                gr.update(value=greeting_text),  # Update greeting
                name  # Store username in state
            )
        
        def ask_question(username, question):
            """Handle question submission"""
            if not question.strip():
                gr.Warning("Please enter a question!")
                return "", ""
            
            classification, answer = process_question(username, question)
            return classification, answer
        
        def clear_inputs():
            """Clear the input fields"""
            return "", "", ""
        
        def reset_session():
            """Reset to username input"""
            return (
                gr.update(visible=True),   # Show username section
                gr.update(visible=False),  # Hide main interface
                gr.update(value=""),       # Clear greeting
                "",  # Clear username state
                "",  # Clear username input
                "",  # Clear question input
                "",  # Clear classification
                ""   # Clear answer
            )
        
        # Event handlers
        start_btn.click(
            start_session,
            inputs=[username_input],
            outputs=[username_row, main_interface, greeting, username_state]
        )
        
        submit_btn.click(
            ask_question,
            inputs=[username_state, question_input],
            outputs=[classification_output, answer_output]
        )
        
        clear_btn.click(
            clear_inputs,
            outputs=[question_input, classification_output, answer_output]
        )
        
        reset_btn.click(
            reset_session,
            outputs=[username_row, main_interface, greeting, username_state, 
                    username_input, question_input, classification_output, answer_output]
        )
        
        # Allow Enter key to submit
        username_input.submit(
            start_session,
            inputs=[username_input],
            outputs=[username_row, main_interface, greeting, username_state]
        )
        
        question_input.submit(
            ask_question,
            inputs=[username_state, question_input],
            outputs=[classification_output, answer_output]
        )
    
    return demo

def main():
    """Launch the Gradio app"""
    print("=== LAUNCHING SMART CODE ASSISTANT WEB APP ===")
    
    demo = create_interface()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=True              # Enable debug mode
    )

if __name__ == "__main__":
    main()