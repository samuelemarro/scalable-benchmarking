import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv
from multiprocessing import Pool

load_dotenv()

from utils import query_llm_batch

PROMPT = """
Provide an answer to the following Math problem:

{QUESTION}

Show the steps you took to arrive at the answer.
"""

def answer_model_worker(args):
    answer_model, all_questions, api_kwargs, temperature = args
    # all_questions: list of (benchmark_model, question_idx, question_text)

    # Group by benchmark_model for output
    answers_by_benchmark = {}
    questions_for_batch = []
    question_keys = []

    for benchmark_model, idx, question in all_questions:
        questions_for_batch.append(question)
        question_keys.append((benchmark_model, idx))

    # Batch query
    try:
        responses = query_llm_batch(answer_model, [PROMPT.format(QUESTION=q) for q in questions_for_batch], temperature=temperature, api_kwargs=api_kwargs)
        # responses should be a list of strings, one per question
    except Exception as e:
        print(e.__class__.__name__, ":", e)
        print("An error occurred during batch query:", e)
        return

    # Save answers to files
    for (benchmark_model, idx), response in zip(question_keys, responses):
        answer_path = Path(f"./answers/{benchmark_model.replace('/','-')}/{answer_model.replace('/','-')}.json")
        answer_path.parent.mkdir(parents=True, exist_ok=True)

        if answer_path.exists() and answer_path.is_file():
            with open(answer_path, "r") as f:
                answer_data = json.load(f)
        else:
            answer_data = {}

        response = response.replace("\\( ", "$").replace("\\(", "$")
        response = response.replace(" \\)", "$").replace("\\)", "$")
        response = response.replace("\\[", "$$").replace("\\]", "$$")

        answer_data[str(idx)] = {
            "question": questions_for_batch[question_keys.index((benchmark_model, idx))],
            "answer": response
        }

        with open(answer_path, "w") as f:
            json.dump(answer_data, f, indent=4)

if __name__ == "__main__":
    reasoning_default = { "effort": "high", "exclude": True }
    models = [
        ("openai/gpt-5-2025-08-07", { "reasoning": reasoning_default }, 0.0),
        ("anthropic/claude-opus-4.1", { "reasoning": reasoning_default }, 1.0),
        ("google/gemini-2.5-pro", { "reasoning": reasoning_default }, 0.0),
        ("openai/gpt-4o-2024-08-06", None, 0.0),
        ("openai/gpt-3.5-turbo", None, 0.0),
        ("meta-llama/llama-4-maverick", None, 0.0),
        ("microsoft/phi-4-reasoning-plus", None, 0.0)
    ]

    # Collect all questions for each answer model (excluding self)
    answer_model_jobs = {answer_model: [] for answer_model, _, _ in models}
    answer_model_kwargs = {answer_model: api_kwargs for answer_model, api_kwargs, _ in models}
    answer_model_temps = {answer_model: temperature for answer_model, _, temperature in models}
    for benchmark_model, _, _ in models:
        benchmark_path = Path(f"./benchmarks/{benchmark_model.replace('/','-')}.json")

        with open(benchmark_path, "r") as f:
            question_data = json.load(f)

        for answer_model, _, _ in models:
            if answer_model == benchmark_model:
                continue
            for i, data in enumerate(question_data):
                answer_path = Path(f"./answers/{benchmark_model.replace('/','-')}/{answer_model.replace('/','-')}.json")
                if answer_path.exists() and answer_path.is_file():
                    with open(answer_path, "r") as f:
                        answer_data = json.load(f)
                else:
                    answer_data = {}

                if str(i) in answer_data:
                    continue

                answer_model_jobs[answer_model].append((benchmark_model, i, data['question']))

    # Pool by answer model
    jobs = [
        (answer_model, answer_model_jobs[answer_model], answer_model_kwargs[answer_model], answer_model_temps[answer_model])
        for answer_model in answer_model_jobs if answer_model_jobs[answer_model]
    ]
    with Pool(processes=len(jobs)) as pool:
        pool.map(answer_model_worker, jobs)
