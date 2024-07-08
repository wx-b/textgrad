import gradio as gr
import textgrad as tg
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss
import io
from PIL import Image
import os

# List of available engines
ENGINES = ["gpt-4o", "gpt-4-vision-preview", "claude-3-opus-20240229"]

def process_image_and_question(image, question, engine, api_key):
    # Set the API key
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Set the backward engine
    tg.set_backward_engine(engine, override=True)

    # Convert uploaded image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Create variables
    image_variable = tg.Variable(img_byte_arr, role_description="image to answer a question about", requires_grad=False)
    question_variable = tg.Variable(question, role_description="question", requires_grad=False)

    # Generate initial response
    response = MultimodalLLMCall(engine)([image_variable, question_variable])

    # Create loss function
    loss_fn = ImageQALoss(
        evaluation_instruction="Does this seem like a complete and good answer for the image? Criticize. Do not provide a new answer.",
        engine=engine
    )

    # Compute loss
    loss = loss_fn(question=question_variable, image=image_variable, response=response)

    # Optimize
    optimizer = tg.TGD(parameters=[response])
    loss.backward()
    optimizer.step()

    return response.value, loss.value, response.value

def run_textgrad_multimodal(image, question, engine, api_key):
    initial_response, loss, optimized_response = process_image_and_question(image, question, engine, api_key)
    return f"Initial Response:\n{initial_response}\n\nLoss:\n{loss}\n\nOptimized Response:\n{optimized_response}"

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# TextGrad Multimodal Optimization Demo")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            question_input = gr.Textbox(label="Enter your question about the image")
            engine_dropdown = gr.Dropdown(choices=ENGINES, label="Select Engine", value="gpt-4o")
            api_key_input = gr.Textbox(label="Enter your API Key", type="password")
            run_button = gr.Button("Run Optimization")
        
        with gr.Column():
            output = gr.Textbox(label="Results", lines=10)

    run_button.click(
        run_textgrad_multimodal,
        inputs=[image_input, question_input, engine_dropdown, api_key_input],
        outputs=output
    )

    gr.Markdown("""
    ## How to use this demo:
    1. Upload an image you want to ask a question about.
    2. Enter your question in the text box.
    3. Select the engine you want to use from the dropdown menu.
    4. Enter your API key (this will be used to authenticate with the selected engine).
    5. Click the "Run Optimization" button to see the results.
    
    The app will display the initial response, the computed loss, and the optimized response.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()