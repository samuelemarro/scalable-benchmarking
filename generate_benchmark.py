import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv
from utils import query_llm_single
from multiprocessing import Pool

load_dotenv()

PROMPT = """
You are to generate ONE very challenging but solvable mathematics problem and its answer.

**Goal.** Create a very hard problem that is difficult for advanced mathematicians, while remaining self-contained.

**Content constraints**
- Must measure genuine mathematical skill (multi-step reasoning, creative insight, careful case control). Avoid ill-posed, trick, or brute-force-only tasks.
- No open problems, no “guess the number”, no huge-number factorization, no tasks requiring external data or long numeric search.
- "Prove that..." problems are allowed.
- Include all needed definitions, assumptions, domains, and bounds.
- Solving the problem should not require tools (e.g., a code interpreter).
- The problem should be as hard as possible while still being solvable.
- Aim for diversity in problem types and topic. You will be provided with a list of questions you have previously written.

**Style/formatting constraints**
- Use GitHub Markdown for the statement and answer (headings, lists, inline math as plain text/LaTeX if useful).
- Do **not** use code blocks in the output.
- Your output should end with the content **exactly** in this order and with these tags (though you are allowed to add more content before this):

[QUESTION]
<problem statement in GitHub Markdown, self-contained, with clearly stated goal and answer format>

[ANSWER]
<the final answer with the steps to reach the solution>



The previous questions you have generated are:
{PREVIOUS_QUESTIONS}

Now generate the problem and answer using the required output format.
"""

def generate_for_model(model, num_samples, reasoning=None):
    internal_model_name = model.replace("/", "-").replace(":", "-")

    path = Path(f'./benchmarks/{internal_model_name}.json')
    path.parent.mkdir(parents=True, exist_ok=True)

    current_data = []
    if path.exists() and path.is_file():
        with open(path, "r") as f:
            current_data = json.load(f)

    for i in range(len(current_data), num_samples):
        try:
            print(f"Generating sample {i + 1} for model {model}...")

            question_list = ""

            if len(current_data) > 0:
                for j, item in enumerate(current_data):
                    if "question" in item:
                        question = item["question"].replace("\n", " ")
                        question_list += f"**{j + 1}.** {question} \n\n"
            else:
                question_list = "No previous questions available."

            specific_prompt = PROMPT.format(PREVIOUS_QUESTIONS=question_list)

            response = query_llm_single(model, specific_prompt, temperature=1, reasoning=reasoning)

            # Split the response into question and answer
            if "[QUESTION]" in response and "[ANSWER]" in response:
                question, answer = response.split("[ANSWER]", 1)
                question = question.replace("[QUESTION]", "").strip()
                answer = answer.strip()
                response = {
                    "question": question,
                    "answer": answer
                }
            else:
                response = {"error": "Response format is incorrect. Expected [QUESTION] and [ANSWER] tags."}

            for k in response.keys():
                response[k] = response[k].replace("\\( ", "$").replace("\\(", "$")
                response[k] = response[k].replace(" \\)", "$").replace("\\)", "$")
                response[k] = response[k].replace("\\[", "$$").replace("\\]", "$$")

            current_data.append(response)

            with open(path, "w") as f:
                json.dump(current_data, f, indent=4)
        except Exception as e:
            print("An error occurred:", e)

if __name__ == "__main__":
    models = [
        ("openai/gpt-5-2025-08-07", "high"),
        ("anthropic/claude-opus-4.1", "high"),
        ("google/gemini-2.5-pro", "high"),
        ("openai/gpt-4o-2024-08-06", None),
        ("openai/gpt-3.5-turbo", None),
        ("meta-llama/llama-4-maverick", None),
        ("microsoft/phi-4-reasoning-plus", None)
    ]
    num_samples = 30

    # Use multiprocessing to parallelize by model
    with Pool(processes=len(models)) as pool:
        pool.starmap(generate_for_model, [(model, num_samples, reasoning) for (model, reasoning) in models])
