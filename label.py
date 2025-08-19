from pathlib import Path
import gradio as gr
import json

latex_delimiters=[{ "left": "$$", "right": "$$", "display": True },
                  { "left": "\\[", "right": "\\]", "display": True },
                  { "left": "\\(", "right": "\\)", "display": False },
                  { "left": "$", "right": "$", "display": False }]

# Hard-coded list of models
models = ["gpt-4o", "gpt-5-mini"]

# Load all model answers
answers = {}

questions = {}

for model in models:
    benchmark_file_path = f'benchmarks/{model}.json'
    if not Path(benchmark_file_path).exists():
        print(f"Benchmark file for {model} does not exist: {benchmark_file_path}")
        continue
    
    with open(f'benchmarks/{model}.json', 'r') as f:
        benchmark_data = json.load(f)
    questions[model] = [q['question'] for q in benchmark_data]
    answers[model] = {}
    answers[model][model] = {}
    for i, data in enumerate(benchmark_data):
        answers[model][model][i] = data['answer']


for question_model in models:
    answers[question_model] = {}
    for answer_model in models:
        file_path = f'answers/{question_model}/{answer_model}.json'
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                answer_data = json.load(f)
            if answer_model not in answers[question_model]:
                answers[question_model][answer_model] = {}
            for key, value in answer_data.items():
                answers[question_model][answer_model][int(key)] = value['answer']

def get_question(idx, question_model):
    idx = max(1, min(idx, len(questions.get(question_model, []))))
    return f"**Question {idx} ({question_model}):**\n\n{questions.get(question_model, [''])[idx-1]}"

def get_answer(idx, question_model, answer_model):
    idx = max(1, min(idx, len(questions.get(question_model, []))))
    a = answers.get(question_model, {}).get(answer_model, {}).get(idx-1, "")
    return f"**Answer ({answer_model}):**\n\n{a}"


# Store user choices: {(question_number, question_model, answer_model): choice}
user_choices = {}
evaluations = {}

def load_evaluations(username):
    eval_path = Path(f"evaluations/{username}.json")
    if eval_path.exists():
        with open(eval_path, "r") as f:
            return json.load(f)
    return {"meaningfulness": {}, "answers": {}}

def save_evaluations(username, evaluations):
    eval_path = Path(f"evaluations/{username}.json")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(evaluations, f, indent=2)

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("## LLM Benchmark Evaluation")
        with gr.Row():
            username_box = gr.Textbox(label="Username", placeholder="Enter your username here")
            confirm_username_btn = gr.Button("Confirm Username")
        error_box = gr.Markdown("", visible=False)

        with gr.Row():
            question_model_dropdown = gr.Dropdown(choices=models, value=models[0], label="Question Model", interactive=False)
            answer_model_dropdown = gr.Dropdown(choices=models, value=models[0], label="Answer Model", interactive=False)
            number = gr.Number(value=1, label="Question Number", interactive=False)

        with gr.Row():
            prev_question_model_button = gr.Button("Previous Question Model", interactive=False)
            next_question_model_button = gr.Button("Next Question Model", interactive=False)
            prev_button = gr.Button("Previous Question", interactive=False)
            next_button = gr.Button("Next Question", interactive=False)
            prev_answer_model_button = gr.Button("Previous Answer Model", interactive=False)
            next_answer_model_button = gr.Button("Next Answer Model", interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                question_markdown = gr.Markdown(get_question(1, models[0]), label="Question", latex_delimiters=latex_delimiters)
                question_well_posed_radio = gr.Radio(
                    choices=["Don't know", "Well-posed", "Not well-posed"],
                    value="Don't know",
                    label="Is the question well-posed?",
                    interactive=False
                )
                question_confidence_slider = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1, label="Confidence in question judgment (1-5)", interactive=False
                )
                question_comment_box = gr.Textbox(label="Comment on question meaningfulness", interactive=False)
            with gr.Column(scale=1):
                answer_markdown = gr.Markdown(get_answer(1, models[0], models[0]), label="Answer", latex_delimiters=latex_delimiters)
                correctness_radio = gr.Radio(
                    choices=["Don't know", "Correct", "Incorrect"],
                    value="Don't know",
                    label="Is the answer correct?",
                    interactive=False
                )
                answer_confidence_slider = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1, label="Confidence in answer judgment (1-5)", interactive=False
                )
                answer_comment_box = gr.Textbox(label="Comment on answer correctness", interactive=False)

        gr.Markdown("---")

    # Helper to enable/disable UI
    def set_ui_enabled(enabled):
        return [
            gr.update(interactive=enabled) for _ in [
                question_model_dropdown, number, answer_model_dropdown,
                prev_question_model_button, next_question_model_button,
                prev_button, next_button,
                question_well_posed_radio, question_confidence_slider, question_comment_box,
                prev_answer_model_button, next_answer_model_button,
                correctness_radio, answer_confidence_slider, answer_comment_box
            ]
        ]

    def update_error(username):
        if not username or username.strip() == "":
            return gr.update(visible=True, value="**Error:** Please enter your username above.")
        return gr.update(visible=False, value="")

    def confirm_username(username):
        if not username or username.strip() == "":
            return set_ui_enabled(False) + [gr.update(visible=True, value="**Error:** Please enter your username above.")]
        # Load evaluations
        global evaluations
        evaluations = load_evaluations(username)
        # Prefill UI for current selection
        idx = number.value
        question_model = question_model_dropdown.value
        answer_model = answer_model_dropdown.value
        question_choice, question_conf, question_comment, answer_choice, answer_conf, answer_comment = prefill_ui(idx, question_model, answer_model, username)
        return (
            *set_ui_enabled(True),
            gr.update(visible=False, value=""),
            # Prefill UI components
            gr.update(value=question_choice),
            gr.update(value=question_conf),
            gr.update(value=question_comment),
            gr.update(value=answer_choice),
            gr.update(value=answer_conf),
            gr.update(value=answer_comment)
        )

    confirm_username_btn.click(
        confirm_username,
        inputs=username_box,
        outputs=[
            question_model_dropdown, number, answer_model_dropdown,
            prev_question_model_button, next_question_model_button,
            prev_button, next_button,
            question_well_posed_radio, question_confidence_slider, question_comment_box,
            prev_answer_model_button, next_answer_model_button,
            correctness_radio, answer_confidence_slider, answer_comment_box,
            error_box,
            question_well_posed_radio, question_confidence_slider, question_comment_box,
            correctness_radio, answer_confidence_slider, answer_comment_box
        ]
    )

    username_box.change(update_error, inputs=username_box, outputs=error_box)

    # Prefill UI from evaluations
    def prefill_ui(idx, question_model, answer_model, username):
        idx = int(idx)
        # Prefill question meaningfulness
        q_eval = evaluations.get("meaningfulness", {}).get(question_model, {}).get(str(idx), {})
        question_choice = q_eval.get("meaningfulness", "Don't know")
        question_conf = q_eval.get("confidence", 3)
        question_comment = q_eval.get("comment", "")
        # Prefill answer correctness
        a_eval = evaluations.get("answers", {}).get(question_model, {}).get(str(idx), {}).get(answer_model, {})
        answer_choice = a_eval.get("correctness", "Don't know")
        answer_conf = a_eval.get("confidence", 3)
        answer_comment = a_eval.get("comment", "")
        return question_choice, question_conf, question_comment, answer_choice, answer_conf, answer_comment

    def save_choice(
        idx, question_model, answer_model,
        answer_choice, answer_conf, question_choice, question_conf,
        answer_comment, question_comment, username
    ):
        if not username or username.strip() == "":
            return gr.update(visible=True, value="**Error:** Please enter your username above.")
        idx = int(idx)
        # Save meaningfulness
        if "meaningfulness" not in evaluations:
            evaluations["meaningfulness"] = {}
        if question_model not in evaluations["meaningfulness"]:
            evaluations["meaningfulness"][question_model] = {}
        evaluations["meaningfulness"][question_model][str(idx)] = {
            "meaningfulness": question_choice,
            "confidence": question_conf,
            "comment": question_comment
        }
        # Save answer, or delete if "Don't know"
        if "answers" not in evaluations:
            evaluations["answers"] = {}
        if question_model not in evaluations["answers"]:
            evaluations["answers"][question_model] = {}
        if str(idx) not in evaluations["answers"][question_model]:
            evaluations["answers"][question_model][str(idx)] = {}
        if answer_choice == "Don't know":
            # Remove entry if exists
            if answer_model in evaluations["answers"][question_model][str(idx)]:
                del evaluations["answers"][question_model][str(idx)][answer_model]
            # If no answers left for this question, remove the question entry
            if not evaluations["answers"][question_model][str(idx)]:
                del evaluations["answers"][question_model][str(idx)]
        else:
            evaluations["answers"][question_model][str(idx)][answer_model] = {
                "correctness": answer_choice,
                "confidence": answer_conf,
                "comment": answer_comment
            }
        save_evaluations(username, evaluations)
        return gr.update(visible=False, value="")

    # Update UI when navigation changes
    def update_question_ui(idx, question_model, answer_model, username):
        question_choice, question_conf, question_comment, answer_choice, answer_conf, answer_comment = prefill_ui(idx, question_model, answer_model, username)
        return (
            get_question(idx, question_model),
            question_choice, question_conf, question_comment,
            get_answer(idx, question_model, answer_model),
            answer_choice, answer_conf, answer_comment
        )

    number.change(
        update_question_ui,
        inputs=[number, question_model_dropdown, answer_model_dropdown, username_box],
        outputs=[
            question_markdown,
            question_well_posed_radio, question_confidence_slider, question_comment_box,
            answer_markdown,
            correctness_radio, answer_confidence_slider, answer_comment_box
        ]
    )
    question_model_dropdown.change(
        update_question_ui,
        inputs=[number, question_model_dropdown, answer_model_dropdown, username_box],
        outputs=[
            question_markdown,
            question_well_posed_radio, question_confidence_slider, question_comment_box,
            answer_markdown,
            correctness_radio, answer_confidence_slider, answer_comment_box
        ]
    )
    answer_model_dropdown.change(
        update_question_ui,
        inputs=[number, question_model_dropdown, answer_model_dropdown, username_box],
        outputs=[
            question_markdown,
            question_well_posed_radio, question_confidence_slider, question_comment_box,
            answer_markdown,
            correctness_radio, answer_confidence_slider, answer_comment_box
        ]
    )

    def next_fn(idx, question_model):
        return min(int(idx) + 1, len(questions.get(question_model, [])))

    def prev_fn(idx):
        return max(int(idx) - 1, 1)

    def next_model_fn(current_model):
        idx = models.index(current_model)
        return models[(idx + 1) % len(models)]

    def prev_model_fn(current_model):
        idx = models.index(current_model)
        return models[(idx - 1) % len(models)]

    # Navigation buttons
    next_button.click(next_fn, inputs=[number, question_model_dropdown], outputs=number)
    prev_button.click(prev_fn, inputs=number, outputs=number)
    next_question_model_button.click(next_model_fn, inputs=question_model_dropdown, outputs=question_model_dropdown)
    prev_question_model_button.click(prev_model_fn, inputs=question_model_dropdown, outputs=question_model_dropdown)
    next_answer_model_button.click(next_model_fn, inputs=answer_model_dropdown, outputs=answer_model_dropdown)
    prev_answer_model_button.click(prev_model_fn, inputs=answer_model_dropdown, outputs=answer_model_dropdown)

    # Save on any change
    correctness_radio.change(
        save_choice,
        inputs=[
            number, question_model_dropdown, answer_model_dropdown,
            correctness_radio, answer_confidence_slider,
            question_well_posed_radio, question_confidence_slider,
            answer_comment_box, question_comment_box,
            username_box
        ],
        outputs=error_box
    )
    answer_confidence_slider.change(
        save_choice,
        inputs=[
            number, question_model_dropdown, answer_model_dropdown,
            correctness_radio, answer_confidence_slider,
            question_well_posed_radio, question_confidence_slider,
            answer_comment_box, question_comment_box,
            username_box
        ],
        outputs=error_box
    )
    answer_comment_box.change(
        save_choice,
        inputs=[
            number, question_model_dropdown, answer_model_dropdown,
            correctness_radio, answer_confidence_slider,
            question_well_posed_radio, question_confidence_slider,
            answer_comment_box, question_comment_box,
            username_box
        ],
        outputs=error_box
    )
    question_well_posed_radio.change(
        save_choice,
        inputs=[
            number, question_model_dropdown, answer_model_dropdown,
            correctness_radio, answer_confidence_slider,
            question_well_posed_radio, question_confidence_slider,
            answer_comment_box, question_comment_box,
            username_box
        ],
        outputs=error_box
    )
    question_confidence_slider.change(
        save_choice,
        inputs=[
            number, question_model_dropdown, answer_model_dropdown,
            correctness_radio, answer_confidence_slider,
            question_well_posed_radio, question_confidence_slider,
            answer_comment_box, question_comment_box,
            username_box
        ],
        outputs=error_box
    )
    question_comment_box.change(
        save_choice,
        inputs=[
            number, question_model_dropdown, answer_model_dropdown,
            correctness_radio, answer_confidence_slider,
            question_well_posed_radio, question_confidence_slider,
            answer_comment_box, question_comment_box,
            username_box
        ],
        outputs=error_box
    )

    def get_question_text(idx, question_model):
        idx = max(1, min(int(idx), len(questions.get(question_model, []))))
        return questions.get(question_model, [""])[idx-1]

    def get_answer_text(idx, question_model, answer_model):
        return answers.get(question_model, {}).get(answer_model, {}).get(int(idx)-1, "")

    # At the end: show original Markdown and copy button
    original_markdown_area = gr.TextArea(label="Original Markdown", value="", interactive=False, lines=10, show_copy_button=True)
    
    def get_original_markdown(idx, question_model, answer_model):
        # Compose the original Markdown for the current question and answer
        idx = max(1, min(int(idx), len(questions.get(question_model, []))))
        q = questions.get(question_model, [""])[idx-1]
        a = answers.get(question_model, {}).get(answer_model, {}).get(idx-1, "")
        return f"**Question:**\n\n{q}\n\n**Answer:**\n\n{a}"

    # Update the textarea whenever navigation changes
    def update_markdown_area(idx, question_model, answer_model):
        return get_original_markdown(idx, question_model, answer_model)

    number.change(update_markdown_area, inputs=[number, question_model_dropdown, answer_model_dropdown], outputs=original_markdown_area)
    question_model_dropdown.change(update_markdown_area, inputs=[number, question_model_dropdown, answer_model_dropdown], outputs=original_markdown_area)
    answer_model_dropdown.change(update_markdown_area, inputs=[number, question_model_dropdown, answer_model_dropdown], outputs=original_markdown_area)

demo.launch()
