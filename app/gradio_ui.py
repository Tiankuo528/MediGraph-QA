import gradio as gr
import requests
import json

# Define the URL of the FastAPI backend
FASTAPI_BACKEND_URL = "http://localhost:8000/generate"

def get_medical_answer(question):
    """
    Sends a question to the FastAPI backend and returns the model's answer.
    """
    if not question:
        return "Please enter a question."

    # The payload to send to the backend
    payload = {"question": question}
    
    try:
        # Make the POST request to the backend
        response = requests.post(FASTAPI_BACKEND_URL, json=payload, timeout=120)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Assuming the backend returns a JSON with a key like "answer" or "generated_text"
            # Adjust the key based on your actual FastAPI response structure
            response_data = response.json()
            # Let's assume the key is 'answer'. Change if necessary.
            return response_data.get("answer", "No answer found in the response.")
        else:
            # Handle non-200 responses
            return f"Error: Received status code {response.status_code}\nResponse: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return "Connection Error: Could not connect to the backend. Please ensure the FastAPI server is running at " + FASTAPI_BACKEND_URL
    except requests.exceptions.Timeout:
        return "Request timed out. The model is taking too long to respond."
    except Exception as e:
        # Handle other potential errors
        return f"An unexpected error occurred: {str(e)}"

# Define the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 MediGraph-QA: Interactive Medical Question Answering
        Ask a medical question below. The system will use a fine-tuned LLaMA3 model and a knowledge graph to generate an answer.
        """
    )
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Your Medical Question", 
            placeholder="e.g., how to identify the tumor bone metastasis from X-day image？",
            lines=3
        )
    
    submit_button = gr.Button("Get Answer", variant="primary")
    
    gr.Markdown("---")
    
    answer_output = gr.Textbox(
        label="Generated Answer", 
        lines=10, 
        interactive=False
    )
    
    # Define the action for the submit button
    submit_button.click(
        fn=get_medical_answer,
        inputs=question_input,
        outputs=answer_output
    )

    gr.Examples(
        examples=[
            "what's the common root cause of the osteoprosis?",
            "What are the symptoms of type 2 diabetes?",
        ],
        inputs=question_input
    )

if __name__ == "__main__":
    # Launch the Gradio web server
    demo.launch(server_name="0.0.0.0", server_port=7860)