import gradio as gr
from huggingface_hub import InferenceClient
import torch
from transformers import pipeline

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Global flag to handle cancellation
stop_inference = False

# Response curation function with multiple generated responses
def respond_curation(
    message,
    system_message="You are a helpful assistant.",
    max_tokens=128,
    temperature=0.7,
    top_p=0.95,
    n_responses=3,  # Number of response options
    use_local_model=False
):
    global stop_inference
    stop_inference = False  # Reset the cancellation flag
    
    responses = []  # List to store all response options
    
    if use_local_model:
        # Local inference
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": message}]
        
        for _ in range(n_responses):
            response = ""
            for output in pipe(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
            ):
                if stop_inference:
                    response = "Inference cancelled."
                    break
                token = output['generated_text'][-1]['content']  # Collect token by token
                response += token
            responses.append(response)  # Store the complete response
            
    else:
        # API-based inference
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": message}]
        
        for _ in range(n_responses):
            response = ""
            # Call the API without expecting a structured 'choices' response
            for message_chunk in client.chat_completion(
                messages,
                max_tokens=max_tokens,
                stream=False,  # Not streaming, generating responses all at once
                temperature=temperature,
                top_p=top_p
            ):
                if isinstance(message_chunk, dict) and "choices" in message_chunk:
                    # When the response is in expected structured format
                    token = message_chunk["choices"][0]["message"]["content"]
                else:
                    # When the response is just a string
                    token = message_chunk
                response += token
            responses.append(response)  # Store the complete response
    
    return responses  # Return all the generated responses

# Function to cancel inference
def cancel_inference():
    global stop_inference
    stop_inference = True

# Vote function for user feedback
def vote(tmp, index_state, data: gr.LikeData):
    value_new = data.value
    index_new = data.index
    if len(index_state) == 0:
        index_state.append(index_new)
    else:
        if index_new in index_state:
            return "Your feedback is already saved", index_state
        else:
            index_state.append(index_new)
    return f"Feedback: {data.value}; Index: {data.index}; Liked: {data.liked}; Votes: {index_state}", index_state

# Custom CSS for a fancy look
custom_css = """
#main-container {
    background-color: #f0f0f0;
    font-family: 'Arial', sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: 0 auto;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
.gr-button {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: #45a049;
}
.gr-slider input {
    color: #4CAF50;
}
.gr-chat {
    font-size: 16px;
}
#title {
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
    color: #333;
}
"""

# Define the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🌟 Fancy AI Chatbot 🌟</h1>")
    gr.Markdown("AI chatbot with customizable settings below.")
    
    with gr.Row():
        system_message = gr.Textbox(value="You are a friendly chatbot.", label="System message", interactive=True)
        use_local_model = gr.Checkbox(label="Use Local Model", value=False)

    with gr.Row():
        max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
        temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
    
    tmp = gr.Textbox(visible=True, value="") 
    chat_history = gr.Chatbot(label="Chat")

    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...")
    
    cancel_button = gr.Button("Cancel Inference", variant="danger")
    index_state = gr.State(value=[])
    
    # Adjusted to ensure history is maintained and passed correctly
    user_input.submit(respond_curation, [user_input, system_message, max_tokens, temperature, top_p, use_local_model], chat_history)
    chat_history.like(vote, [tmp, index_state], [tmp, index_state])
    cancel_button.click(cancel_inference)

# Launch the demo
if __name__ == "__main__":
    demo.launch(share=False)
