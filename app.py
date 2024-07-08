import gradio as gr
import textgrad as tg
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss, TextLoss
import io
from PIL import Image
import os

# List of available engines
ENGINES = ["gpt-4o", "gpt-4-vision-preview", "claude-3-opus-20240229"]

def process_image_and_question(image, question, engine, api_key, evaluation_instruction, iterations):
    os.environ["OPENAI_API_KEY"] = api_key
    tg.set_backward_engine(engine, override=True)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    image_variable = tg.Variable(img_byte_arr, role_description="image to answer a question about", requires_grad=False)
    question_variable = tg.Variable(question, role_description="question", requires_grad=False)

    # Initial response
    initial_response = MultimodalLLMCall(engine)([image_variable, question_variable])

    # Create a variable for the response that we can optimize
    response = tg.Variable(initial_response.value, role_description="response to the question", requires_grad=True)

    loss_fn = ImageQALoss(
        evaluation_instruction=evaluation_instruction,
        engine=engine
    )

    # Optimization loop
    for _ in range(iterations):
        loss = loss_fn(question=question_variable, image=image_variable, response=response)
        optimizer = tg.TGD(parameters=[response])
        loss.backward()
        optimizer.step()

    return initial_response.value, str(loss.value), response.value

def run_textgrad_multimodal(image, question, engine, api_key, evaluation_instruction, iterations):
    initial_response, loss, optimized_response = process_image_and_question(image, question, engine, api_key, evaluation_instruction, iterations)
    return initial_response, loss, optimized_response

def process_solution_optimization(initial_input, role_description, loss_prompt, engine, api_key, iterations):
    os.environ["OPENAI_API_KEY"] = api_key
    tg.set_backward_engine(engine, override=True)

    variable = tg.Variable(initial_input, requires_grad=True, role_description=role_description)

    loss_system_prompt = tg.Variable(loss_prompt, requires_grad=False, role_description="system prompt")
                              
    loss_fn = TextLoss(loss_system_prompt)
    optimizer = tg.TGD([variable])

    # Optimization loop
    for _ in range(iterations):
        loss = loss_fn(variable)
        loss.backward()
        optimizer.step()

    return initial_input, str(loss.value), variable.value

def run_solution_optimization(initial_input, role_description, loss_prompt, engine, api_key, iterations):
    initial, loss, optimized = process_solution_optimization(initial_input, role_description, loss_prompt, engine, api_key, iterations)
    return initial, loss, optimized

def process_prompt_optimization(system_prompt, input_prompt, question, answer, engine, api_key, iterations):
    os.environ["OPENAI_API_KEY"] = api_key
    tg.set_backward_engine(engine, override=True)

    # Create a dummy evaluation function
    def dummy_eval_fn(inputs):
        prediction = inputs['prediction'].value
        ground_truth = inputs['ground_truth_answer'].value
        # Simple string comparison for demonstration purposes
        score = 1 if prediction.strip().lower() == ground_truth.strip().lower() else 0
        return tg.Variable(str(score), requires_grad=False, role_description="evaluation score")

    question = tg.Variable(question, role_description="question to the LLM", requires_grad=False)
    answer = tg.Variable(answer, role_description="answer to the question", requires_grad=False)

    system_prompt = tg.Variable(system_prompt,
                                requires_grad=True,
                                role_description="system prompt to guide the LLM's reasoning strategy for accurate responses")

    input_prompt = tg.Variable(input_prompt,
                               requires_grad=True,
                               role_description="input prompt for the LLM")

    model = tg.BlackboxLLM(engine, system_prompt=system_prompt)
    optimizer = tg.TGD(parameters=list(model.parameters()) + [input_prompt])

    # Optimization loop
    for _ in range(iterations):
        full_prompt = tg.Variable(f"{input_prompt.value}\n\n{question.value}", requires_grad=True, role_description="full prompt for the LLM")
        prediction = model(full_prompt)
        loss = dummy_eval_fn(dict(prediction=prediction, ground_truth_answer=answer))
        loss.backward()
        optimizer.step()

    return system_prompt.value, input_prompt.value, prediction.value, str(loss.value)

def run_prompt_optimization(system_prompt, input_prompt, question, answer, engine, api_key, iterations):
    optimized_system_prompt, optimized_input_prompt, prediction, loss = process_prompt_optimization(system_prompt, input_prompt, question, answer, engine, api_key, iterations)
    return optimized_system_prompt, optimized_input_prompt, prediction, loss

def process_code_optimization(problem_text, initial_solution, engine, api_key, iterations):
    os.environ["OPENAI_API_KEY"] = api_key
    tg.set_backward_engine(engine, override=True)

    llm_engine = tg.get_engine(engine)

    code = tg.Variable(value=initial_solution,
                       requires_grad=True,
                       role_description="code instance to optimize")

    problem = tg.Variable(problem_text, 
                          requires_grad=False, 
                          role_description="the coding problem")

    optimizer = tg.TGD(parameters=[code])

    loss_system_prompt = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise feedback."
    loss_system_prompt = tg.Variable(loss_system_prompt, requires_grad=False, role_description="system prompt to the loss function")

    instruction = """Think about the problem and the code snippet. Does the code solve the problem? What is the runtime complexity?"""
    format_string = "{instruction}\nProblem: {{problem}}\nCurrent Code: {{code}}"
    format_string = format_string.format(instruction=instruction)

    fields = {"problem": None, "code": None}
    formatted_llm_call = tg.autograd.FormattedLLMCall(engine=llm_engine,
                                          format_string=format_string,
                                          fields=fields,
                                          system_prompt=loss_system_prompt)

    def loss_fn(problem: tg.Variable, code: tg.Variable) -> tg.Variable:
        inputs = {"problem": problem, "code": code}
        return formatted_llm_call(inputs=inputs,
                                  response_role_description=f"evaluation of the {code.get_role_description()}")

    # Optimization loop
    for _ in range(iterations):
        loss = loss_fn(problem, code)
        loss.backward()
        optimizer.step()

    return initial_solution, str(loss.value), code.value

def run_code_optimization(problem_text, initial_solution, engine, api_key, iterations):
    initial, loss, optimized = process_code_optimization(problem_text, initial_solution, engine, api_key, iterations)
    return initial, loss, optimized

def check_api_key(api_key):
    if not api_key:
        return gr.Warning("Please enter your API key before running optimizations.")

def update_selected_outputs(*args):
    checkboxes = args[:len(args)//2]
    outputs = args[len(args)//2:]
    selected = ""
    labels = ["Uploaded Image", "Question", "Evaluation Instruction", "Initial Response", "Loss", "Optimized Response", 
              "Initial Input", "Role Description", "Loss Prompt", "Optimized Solution", 
              "System Prompt", "Input Prompt", "Question", "Answer", "Optimized System Prompt", "Optimized Input Prompt", "Prediction",
              "Problem Description", "Initial Code", "Optimized Code"]

    for checkbox, output, label in zip(checkboxes, outputs, labels):
        if checkbox and output:
            selected += f"{label}:\n{output}\n\n"
    return selected


with gr.Blocks() as demo:
    gr.Markdown("# TextGrad Optimization Demo")
    
    with gr.Row():
        with gr.Column(scale=1):
            engine_dropdown = gr.Dropdown(choices=ENGINES, label="Select Engine", value="gpt-4o")
            iterations = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Number of Optimization Iterations")
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(label="Enter your API Key", type="password")

    gr.Markdown("---")

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
                    run_button = gr.Button("Run Optimization")
                
                with gr.Column():
                    initial_response = gr.Textbox(label="Initial Response", lines=5)
                    loss_output = gr.Textbox(label="Loss", lines=2)
                    optimized_response = gr.Textbox(label="Optimized Response", lines=5)

            run_button.click(
                run_textgrad_multimodal,
                inputs=[image_input, question_input, engine_dropdown, api_key_input, evaluation_instruction, iterations],
                outputs=[initial_response, loss_output, optimized_response]
            )

            gr.Markdown("## Select outputs to download:")
            with gr.Row():
                multimodal_image_checkbox = gr.Checkbox(label="Uploaded Image", value=False)
                multimodal_question_checkbox = gr.Checkbox(label="Question", value=False)
                multimodal_evaluation_checkbox = gr.Checkbox(label="Evaluation Instruction", value=False)
                multimodal_initial_checkbox = gr.Checkbox(label="Initial Response", value=False)
                multimodal_loss_checkbox = gr.Checkbox(label="Loss", value=False)
                multimodal_optimized_checkbox = gr.Checkbox(label="Optimized Response", value=True)

            multimodal_selected_outputs = gr.Code(label="Selected Outputs", language="markdown", lines=10)

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown(" ")
                with gr.Column(scale=1):
                    multimodal_show_button = gr.Button("Show Selected")

            multimodal_show_button.click(
                update_selected_outputs,
                inputs=[multimodal_image_checkbox, multimodal_question_checkbox, multimodal_evaluation_checkbox, multimodal_initial_checkbox, multimodal_loss_checkbox, multimodal_optimized_checkbox,
                        image_input, question_input, evaluation_instruction, initial_response, loss_output, optimized_response],
                outputs=[multimodal_selected_outputs]
            )


        with gr.TabItem("Solution Optimization"):
            with gr.Row():
                with gr.Column():
                    initial_input = gr.Textbox(label="Enter initial input", lines=3, value="write a punchline for my github package about saving pages and documents to collections for llm contexts")
                    role_description = gr.Textbox(label="Role Description", value="a concise punchline that must hook everyone")
                    loss_prompt = gr.Textbox(label="Loss Function Prompt", lines=3, value="We want to have a super smart and funny punchline. Is the current one concise and addictive? Is the punch fun, makes sense, and subtle enough?")
                    run_button_sol = gr.Button("Run Solution Optimization")
                
                with gr.Column():
                    initial_solution_output = gr.Textbox(label="Initial Input", lines=3)
                    loss_output_sol = gr.Textbox(label="Loss", lines=2)
                    optimized_solution = gr.Textbox(label="Optimized Solution", lines=3)

            run_button_sol.click(
                run_solution_optimization,
                inputs=[initial_input, role_description, loss_prompt, engine_dropdown, api_key_input, iterations],
                outputs=[initial_solution_output, loss_output_sol, optimized_solution]
            )

            gr.Markdown("## Select outputs to download:")
            with gr.Row():
                solution_initial_input_checkbox = gr.Checkbox(label="Initial Input", value=False)
                solution_role_description_checkbox = gr.Checkbox(label="Role Description", value=False)
                solution_loss_prompt_checkbox = gr.Checkbox(label="Loss Prompt", value=False)
                solution_initial_checkbox = gr.Checkbox(label="Initial Input", value=False)
                solution_loss_checkbox = gr.Checkbox(label="Loss", value=False)
                solution_optimized_checkbox = gr.Checkbox(label="Optimized Solution", value=True)

            solution_selected_outputs = gr.Code(label="Selected Outputs", language="markdown", lines=10)

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown(" ")
                with gr.Column(scale=1):
                    solution_show_button = gr.Button("Show Selected")


            solution_show_button.click(
                update_selected_outputs,
                inputs=[solution_initial_input_checkbox, solution_role_description_checkbox, solution_loss_prompt_checkbox, solution_initial_checkbox, solution_loss_checkbox, solution_optimized_checkbox,
                        initial_input, role_description, loss_prompt, initial_solution_output, loss_output_sol, optimized_solution],
                outputs=[solution_selected_outputs]
            )


        with gr.TabItem("Prompt Optimization"):
            with gr.Row():
                with gr.Column():
                    system_prompt_input = gr.Textbox(label="System Prompt", lines=3, value="You are a concise LLM. Think step by step.")
                    input_prompt_input = gr.Textbox(label="Input Prompt", lines=3, value="Count the objects in the following question:")
                    question_input_prompt = gr.Textbox(label="Question", lines=3, value="I have two stalks of celery, two garlics, a potato, three heads of broccoli, a carrot, and a yam. How many vegetables do I have?")
                    answer_input_prompt = gr.Textbox(label="Answer", lines=1, value="10")
                    run_button_prompt = gr.Button("Run Prompt Optimization")
                
                with gr.Column():
                    optimized_system_prompt_output = gr.Textbox(label="Optimized System Prompt", lines=3)
                    optimized_input_prompt_output = gr.Textbox(label="Optimized Input Prompt", lines=3)
                    prediction_output = gr.Textbox(label="Prediction", lines=3)
                    loss_output_prompt = gr.Textbox(label="Loss", lines=2)

            run_button_prompt.click(
                run_prompt_optimization,
                inputs=[system_prompt_input, input_prompt_input, question_input_prompt, answer_input_prompt, engine_dropdown, api_key_input, iterations],
                outputs=[optimized_system_prompt_output, optimized_input_prompt_output, prediction_output, loss_output_prompt]
            )

            gr.Markdown("## Select outputs to download:")
            with gr.Row():
                prompt_system_prompt_checkbox = gr.Checkbox(label="System Prompt", value=True)
                prompt_input_prompt_checkbox = gr.Checkbox(label="Input Prompt", value=True)
                prompt_question_checkbox = gr.Checkbox(label="Question", value=False)
                prompt_answer_checkbox = gr.Checkbox(label="Answer", value=False)
                prompt_system_checkbox = gr.Checkbox(label="Optimized System Prompt", value=True)
                prompt_input_checkbox = gr.Checkbox(label="Optimized Input Prompt", value=True)
                prompt_prediction_checkbox = gr.Checkbox(label="Prediction", value=False)
                prompt_loss_checkbox = gr.Checkbox(label="Loss", value=False)

            prompt_selected_outputs = gr.Code(label="Selected Outputs", language="markdown", lines=10)

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown(" ")
                with gr.Column(scale=1):
                    prompt_show_button = gr.Button("Show Selected")


            prompt_show_button.click(
                update_selected_outputs,
                inputs=[prompt_system_prompt_checkbox, prompt_input_prompt_checkbox, prompt_question_checkbox, prompt_answer_checkbox, prompt_system_checkbox, prompt_input_checkbox, prompt_prediction_checkbox, prompt_loss_checkbox,
                        system_prompt_input, input_prompt_input, question_input_prompt, answer_input_prompt, optimized_system_prompt_output, optimized_input_prompt_output, prediction_output, loss_output_prompt],
                outputs=[prompt_selected_outputs]
            )


        with gr.TabItem("Code Optimization"):
            with gr.Row():
                with gr.Column():
                    problem_text_input = gr.Textbox(label="Problem Description", lines=5, value="""Longest Increasing Subsequence (LIS)

Problem Statement:
Given a sequence of integers, find the length of the longest subsequence that is strictly increasing. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

Input:
The input consists of a list of integers representing the sequence.

Output:
The output should be an integer representing the length of the longest increasing subsequence.""")
                    initial_code_input = gr.Code(label="Initial Code", language="python", lines=10, value="""
def longest_increasing_subsequence(nums):
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    max_length = max(dp)
    lis = []
    
    for i in range(n - 1, -1, -1):
        if dp[i] == max_length:
            lis.append(nums[i])
            max_length -= 1
    
    return len(lis[::-1])
""")
                    run_button_code = gr.Button("Run Code Optimization")
                
                with gr.Column():
                    initial_code_output = gr.Code(label="Initial Code", language="python", lines=10)
                    loss_output_code = gr.Textbox(label="Loss", lines=2)
                    optimized_code_output = gr.Code(label="Optimized Code", language="python", lines=10)

            run_button_code.click(
                run_code_optimization,
                inputs=[problem_text_input, initial_code_input, engine_dropdown, api_key_input, iterations],
                outputs=[initial_code_output, loss_output_code, optimized_code_output]
            )

            gr.Markdown("## Select outputs to download:")
            with gr.Row():
                code_problem_text_checkbox = gr.Checkbox(label="Problem Description", value=False)
                code_initial_solution_checkbox = gr.Checkbox(label="Initial Code", value=False)
                code_loss_checkbox = gr.Checkbox(label="Loss", value=False)
                code_optimized_checkbox = gr.Checkbox(label="Optimized Code", value=True)

            code_selected_outputs = gr.Code(label="Selected Outputs", language="markdown", lines=10)

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown(" ")
                with gr.Column(scale=1):
                    code_show_button = gr.Button("Show Selected")


            code_show_button.click(
                update_selected_outputs,
                inputs=[code_problem_text_checkbox, code_initial_solution_checkbox, code_loss_checkbox, code_optimized_checkbox,
                        problem_text_input, initial_code_input, loss_output_code, optimized_code_output],
                outputs=[code_selected_outputs]
            )


    gr.Markdown("""
    ## How to use this demo:
    1. Enter your API key at the top of the page.
    2. Select the engine you want to use from the dropdown menu.
    3. Choose the number of optimization iterations using the slider.
    4. Choose the tab for the type of optimization you want to perform.
    5. After running optimizations, select which outputs you want to download using the checkboxes.
    6. Click the "Show Selected" button to display the selected outputs.
    
    ### Multimodal Optimization Tab:
    1. Upload an image you want to ask a question about.
    2. Enter your question in the text box.
    3. Modify the evaluation instruction if needed.
    4. Click the "Run Optimization" button to see the results.

    ### Solution Optimization Tab:
    1. Enter the initial input (e.g., a punchline or text to optimize).
    2. Specify the role description for the variable.
    3. Enter the loss function prompt.
    4. Click the "Run Solution Optimization" button to see the results.

    ### Prompt Optimization Tab:
    1. Enter the system prompt and input prompt you want to optimize.
    2. Provide a sample question and answer for evaluation.
    3. Click the "Run Prompt Optimization" button to see the results.

    ### Code Optimization Tab:
    1. Enter the problem description in the text box.
    2. Provide the initial code solution in the code editor.
    3. Click the "Run Code Optimization" button to see the results.

    """)

    for run_button in [run_button, run_button_sol, run_button_prompt, run_button_code]:
        run_button.click(check_api_key, inputs=[api_key_input])

if __name__ == "__main__":
    demo.launch()
