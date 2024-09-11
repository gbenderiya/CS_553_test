import gradio as gr
from huggingface_hub import InferenceClient
import torch
from transformers import pipeline

# Inference client setup
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
pipe = pipeline("text-generation", "microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16, device_map="auto")

# Global flag to handle cancellation
stop_inference = False

personas = {
    "Friendly": "You are a friendly and approachable chatbot.",
    "Formal": "You are a professional and formal chatbot.",
    "Humorous": "You are a chatbot with a humorous and playful tone.",
    "Concise": "You are a chatbot that provides concise and to-the-point responses.",
    "Detailed": "You are a chatbot that provides detailed and thorough explanations."
}

def adjust_temperature(message: str) -> float:

    """

    Adjust the temperature dynamically based on the message content.

    - Lower temperature for factual questions.

    - Higher temperature for creative or brainstorming responses.

    """

    keywords_for_factual = ["what", "who", "when", "where", "explain", "define"]

    keywords_for_creative = ["imagine", "brainstorm", "create", "idea", "suggest"]


    # Lower temperature for short or factual queries

    if any(keyword in message.lower() for keyword in keywords_for_factual) or len(message.split()) < 5:

        return 0.3  # Factual, concise

    # Higher temperature for open-ended or creative queries

    elif any(keyword in message.lower() for keyword in keywords_for_creative) or len(message.split()) > 15:

        return 0.9  # Creative, open-ended

    # Default temperature for general queries

    return 0.7

def respond(
    message,
    history: list[tuple[str, str]],
    system_message=None,
    max_tokens=512,
    temperature= None,
    top_p=0.95,
    use_local_model=False,
    persona = 'Friendly'
):
    global stop_inference
    stop_inference = False  # Reset cancellation flag

    # Initialize history if it's None
    if history is None:
        history = []
    
    dynamic_temperature = adjust_temperature(message) if temperature is None else temperature
    
    if system_message is None:
        system_message = personas.get(persona, "You are a friendly Chatbot.")

    if use_local_model:
        # local inference 
        messages = [{"role": "system", "content": system_message}]
        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})
        messages.append({"role": "user", "content": message})

        response = ""
        for output in pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=dynamic_temperature,
            do_sample=True,
            top_p=top_p,
        ):
            if stop_inference:
                response = "Inference cancelled."
                yield history + [(message, response)]
                return
            token = output['generated_text'][-1]['content']
            response += token
            yield history + [(message, response)]  # Yield history + new response

    else:
        # API-based inference 
        messages = [{"role": "system", "content": system_message}]
        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})
        messages.append({"role": "user", "content": message})

        response = ""
        for message_chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=dynamic_temperature,
            top_p=top_p,
        ):
            if stop_inference:
                response = "Inference cancelled."
                yield history + [(message, response)]
                return
            if stop_inference:
                response = "Inference cancelled."
                break
            token = message_chunk.choices[0].delta.content
            response += token
            yield history + [(message, response)]  # Yield history + new response


def cancel_inference():
    global stop_inference
    stop_inference = True

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

def update_system_message(persona):
    return personas.get(persona, "You are a friendly Chatbot.")


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

# Define the interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🌟 Fancy AI Chatbot 🌟</h1>")
    gr.Markdown("AI chatbot using customizable settings below.  kkkkk")
    

    with gr.Row():
        system_message = gr.Textbox(value="You are a friendly Chatbot.", label="System message", interactive=True)
        use_local_model = gr.Checkbox(label="Use Local Model", value=False)
        persona = gr.Dropdown(choices=list(personas.keys()), value="Friendly", label="Select Persona")


    with gr.Row():
        max_tokens = gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens")
        temperature = gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature", visible = False)
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
    
    tmp = gr.Textbox(visible=True, value="") 
    chat_history = gr.Chatbot(label="Chat")

    user_input = gr.Textbox(show_label=False, placeholder="Type your message here...")
    
    cancel_button = gr.Button("Cancel Inference", variant="danger")
    index_state = gr.State(value=[])

    # Submit function to handle user input
    def submit_function(message, history, system_message, max_tokens, temperature, top_p, use_local_model, persona):
        system_message = update_system_message(persona)
        return respond(message, history, system_message, max_tokens, temperature, top_p, use_local_model, persona)
 
    user_input.submit(submit_function, [user_input, chat_history, system_message, max_tokens, temperature, top_p, use_local_model, persona], chat_history)
   
    chat_history.like(vote, [tmp, index_state], [tmp, index_state])
    cancel_button.click(cancel_inference)

if __name__ == "__main__":
    demo.launch(share=False)