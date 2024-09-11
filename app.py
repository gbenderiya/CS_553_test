import gradio as gr
from huggingface_hub import InferenceClient
import torch
from transformers import pipeline

# Set up the inference pipeline
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Function to generate multiple responses
def respond_curation(
    message,
    system_message="You are a helpful assistant.",
    max_tokens=128,
    temperature=0.7,
    top_p=0.95,
    n_responses=3,  # Number of response options
    use_local_model=False
):
    responses = []  # List to store all response options
    if use_local_model:
        # Local inference
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": message}]
        
        for _ in range(n_responses):
            generated_response = ""
            for output in pipe(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p
            ):
                generated_response += output["generated_text"]
            responses.append(generated_response)
    else:
        # API-based inference
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": message}]
        
        for _ in range(n_responses):
            generated_response = ""
            for message_chunk in client.chat_completion(
                messages,
                max_tokens=max_tokens,
                stream=False,  # Set to False since we are curating multiple responses
                temperature=temperature,
                top_p=top_p
            ):
                generated_response += message_chunk.choices[0].message["content"]
            responses.append(generated_response)
    
    return responses  # Return all the generated responses

# Custom CSS for styling
custom_css = """
#main-container {
    background-color: #f9f9f9;
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
"""

# Define the Gradio interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>💬 Chatbot with Response Curation 💬</h1>")
    
    system_message = gr.Textbox(value="You are a helpful assistant.", label="System message")
    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...")
    
    max_tokens = gr.Slider(minimum=1, maximum=512, value=128, step=1, label="Max new tokens")
    temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
    top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
    n_responses = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Number of Responses")
    
    use_local_model = gr.Checkbox(label="Use Local Model", value=False)
    chat_history = gr.Chatbot(label="Chat")

    response_options = gr.Radio(label="Select a response", choices=[], visible=False)
    selected_response = gr.Textbox(label="Selected response will appear here")

    def generate_responses(message, system_message, max_tokens, temperature, top_p, n_responses, use_local_model):
        responses = respond_curation(
            message=message,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n_responses=n_responses,
            use_local_model=use_local_model
        )
        response_options.update(choices=responses, visible=True)
        return "", responses

    def select_response(selected):
        return f"You selected: {selected}"

    user_input.submit(generate_responses, 
                      [user_input, system_message, max_tokens, temperature, top_p, n_responses, use_local_model],
                      [user_input, response_options])
    
    response_options.change(select_response, response_options, selected_response)

# Launch the Gradio demo
if __name__ == "__main__":
    demo.launch(share=False)
