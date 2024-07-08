import gradio as gr
import textgrad as tg
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss, TextLoss
import io
from PIL import Image
import os

# List of available engines
ENGINES = ["gpt-4o", "gpt-4-vision-preview", "claude-3-opus-20240229"]

def process_image_and_question(image, question, engine, api_key, evaluation_instruction):
    os.environ["OPENAI_API_KEY"] = api_key
    tg.set_backward_engine(engine, override=True)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    image_variable = tg.Variable(img_byte_arr, role_description="image to answer a question about", requires_grad=False)
    question_variable = tg.Variable(question, role_description="question", requires_grad=False)

    response = MultimodalLLMCall(engine)([image_variable, question_variable])

    loss_fn = ImageQALoss(
        evaluation_instruction=evaluation_instruction,
        engine=engine
    )

    loss = loss_fn(question=question_variable, image=image_variable, response=response)

    optimizer = tg.TGD(parameters=[response])
    loss.backward()
    optimizer.step()

    return response.value, str(loss.value), response.value

def run_textgrad_multimodal(image, question, engine, api_key, evaluation_instruction):
    initial_response, loss, optimized_response = process_image_and_question(image, question, engine, api_key, evaluation_instruction)
    return initial_response, loss, optimized_response

def process_solution_optimization(initial_input, role_description, loss_prompt, engine, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    tg.set_backward_engine(engine, override=True)

    variable = tg.Variable(initial_input, requires_grad=True, role_description=role_description)

    loss_system_prompt = tg.Variable(loss_prompt, requires_grad=False, role_description="system prompt")
                              
    loss_fn = TextLoss(loss_system_prompt)
    optimizer = tg.TGD([variable])

    loss = loss_fn(variable)
    loss.backward()
    optimizer.step()

    return initial_input, str(loss.value), variable.value

def run_solution_optimization(initial_input, role_description, loss_prompt, engine, api_key):
    initial, loss, optimized = process_solution_optimization(initial_input, role_description, loss_prompt, engine, api_key)
    return initial, loss, optimized

with gr.Blocks() as demo:
    gr.Markdown("# TextGrad Optimization Demo")
    
    with gr.Tabs():
        with gr.TabItem("Multimodal Optimization"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Image")
                    question_input = gr.Textbox(label="Enter your question about the image")
                    evaluation_instruction = gr.Textbox(
                        label="Evaluation Instruction",
                        value="Does this seem like a complete and good answer for the image? Criticize. Do not provide a new answer.",
                        lines=3
                    )
                    engine_dropdown = gr.Dropdown(choices=ENGINES, label="Select Engine", value="gpt-4o")
                    api_key_input = gr.Textbox(label="Enter your API Key", type="password")
                    run_button = gr.Button("Run Optimization")
                
                with gr.Column():
                    initial_response = gr.Textbox(label="Initial Response", lines=5)
                    loss_output = gr.Textbox(label="Loss", lines=2)
                    optimized_response = gr.Textbox(label="Optimized Response", lines=5)
                    copy_button = gr.Button("Copy Results")

            run_button.click(
                run_textgrad_multimodal,
                inputs=[image_input, question_input, engine_dropdown, api_key_input, evaluation_instruction],
                outputs=[initial_response, loss_output, optimized_response]
            )

        with gr.TabItem("Solution Optimization"):
            with gr.Row():
                with gr.Column():
                    initial_input = gr.Textbox(label="Enter initial input", lines=3, value="write a punchline for my github package about saving pages and documents to collections for llm contexts")
                    role_description = gr.Textbox(label="Role Description", value="a concise punchline that must hook everyone")
                    loss_prompt = gr.Textbox(label="Loss Function Prompt", lines=3, value="We want to have a super smart and funny punchline. Is the current one concise and addictive? Is the punch fun, makes sense, and subtle enough?")
                    engine_dropdown_sol = gr.Dropdown(choices=ENGINES, label="Select Engine", value="gpt-4o")
                    api_key_input_sol = gr.Textbox(label="Enter your API Key", type="password")
                    run_button_sol = gr.Button("Run Solution Optimization")
                
                with gr.Column():
                    initial_solution_output = gr.Textbox(label="Initial Input", lines=3)
                    loss_output_sol = gr.Textbox(label="Loss", lines=2)
                    optimized_solution = gr.Textbox(label="Optimized Solution", lines=3)
                    copy_button_sol = gr.Button("Copy Results")

            run_button_sol.click(
                run_solution_optimization,
                inputs=[initial_input, role_description, loss_prompt, engine_dropdown_sol, api_key_input_sol],
                outputs=[initial_solution_output, loss_output_sol, optimized_solution]
            )

    gr.Markdown("""
    ## How to use this demo:
    ### Multimodal Optimization Tab:
    1. Upload an image you want to ask a question about.
    2. Enter your question in the text box.
    3. Modify the evaluation instruction if needed.
    4. Select the engine you want to use from the dropdown menu.
    5. Enter your API key.
    6. Click the "Run Optimization" button to see the results.

    ### Solution Optimization Tab:
    1. Enter the initial input (e.g., a punchline or text to optimize).
    2. Specify the role description for the variable.
    3. Enter the loss function prompt.
    4. Select the engine you want to use from the dropdown menu.
    5. Enter your API key.
    6. Click the "Run Solution Optimization" button to see the results.

    You can copy the results using the "Copy Results" button in each tab. The results will be copied to your clipboard.
    """)

    # Embed JavaScript for copying text to clipboard
    gr.HTML("""
    <script>
        function copyTextToClipboard(text) {
            navigator.clipboard.writeText(text).then(function() {
                console.log('Copying to clipboard was successful!');
            }, function(err) {
                console.error('Could not copy text: ', err);
            });
        }

        function setupCopyButtons() {
            const copyButtons = document.querySelectorAll('button:contains("Copy Results")');
            copyButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const container = button.closest('.tabitem');
                    const textareas = container.querySelectorAll('textarea');
                    const copyText = Array.from(textareas).map(textarea => `${textarea.previousElementSibling.textContent}:\n${textarea.value}`).join('\n\n');
                    copyTextToClipboard(copyText);
                });
            });
        }

        // Use MutationObserver to handle dynamically added elements
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList') {
                    setupCopyButtons();
                }
            });
        });

        observer.observe(document.body, { childList: true, subtree: true });

        window.addEventListener('load', setupCopyButtons);
    </script>
    """)

if __name__ == "__main__":
    demo.launch()