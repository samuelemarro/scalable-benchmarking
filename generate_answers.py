import os
import requests
import json

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from utils import query_llm_single

PROMPT = """
Provide an answer to the following Math problem::

{QUESTION}

Show the steps you took to arrive at the answer.
"""

if __name__ == "__main__":
    models = ["gpt-4o", "gpt-5-mini"]

    for benchmark_model in models:
        benchmark_path = Path(f'./benchmarks/{benchmark_model}.json')
        with open(benchmark_path, "r") as f:
            question_data = json.load(f)

        for answer_model in models:
            if benchmark_model == answer_model:
                continue

            answer_path = Path(f'./answers/{benchmark_model}/{answer_model}.json')

            answer_path.parent.mkdir(parents=True, exist_ok=True)

            if answer_path.exists() and answer_path.is_file():
                with open(answer_path, "r") as f:
                    answer_data = json.load(f)
            else:
                answer_data = {}

            for i, data in enumerate(question_data):
                if str(i) in answer_data:
                    print(f"Skipping question {i+1}/{len(question_data)}: {data['question']} (already answered)")
                    continue
                print(f"{answer_model} answering {question_model} question {i+1}/{len(question_data)}: {data['question']}")

                question = data['question']
                try:
                    response = query_llm_single(answer_model, PROMPT.format(QUESTION=question), temperature=0)

                    response = response.replace("\\( ", "$").replace("\\(", "$")
                    response = response.replace(" \\)", "$").replace("\\)", "$")
                    response = response.replace("\\[", "$$").replace("\\]", "$$")

                    answer_data[str(i)] = {
                        "question": question,
                        "answer": response
                    }

                    with open(answer_path, "w") as f:
                        json.dump(answer_data, f, indent=4)

                except Exception as e:
                    print(e.__class__.__name__, ":", e)
                    print("An error occurred:", e)
