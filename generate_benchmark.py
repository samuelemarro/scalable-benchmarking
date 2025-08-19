import os
import requests
import json

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from utils import query_llm_single

PROMPT = """
You are to generate ONE very challenging but fair mathematics problem and its answer.

**Goal.** Create a very hard problem that is difficult for advanced solvers and especially challenging for LLMs, while remaining self-contained and verifiable.

**Content constraints**
- Must measure genuine mathematical skill (multi-step reasoning, creative insight, careful case control). Avoid ill-posed, trick, or brute-force-only tasks.
- No open problems, no “guess the number”, no huge-number factorization, no tasks requiring external data or long numeric search.
- Proofs are allowed.
- Include all needed definitions, assumptions, domains, and bounds.
- Solving the problem should not require tools (e.g., a code interpreter).
- The problem should be as hard as possible while still being solvable.

**Style/formatting constraints**
- Use GitHub Markdown for the statement and answer (headings, lists, inline math as plain text/LaTeX if useful).
- Do **not** use code blocks in the output.
- Your output should end with the content **exactly** in this order and with these tags:

[QUESTION]
<problem statement in GitHub Markdown, self-contained, with clearly stated goal and answer format>

[ANSWER]
<the final answer with the steps to reach the solution> 

Now generate the problem and answer using the required output format.
"""

if __name__ == "__main__":
    models = ["gpt-4o", "gpt-5-mini"]
    num_samples = 10
    for model in models:
        for i in range(num_samples):
            try:
                response = query_llm_single(model, PROMPT)

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

                # Save everything in a JSON file
                path = Path(f'./benchmarks/{model}.json')
                path.parent.mkdir(parents=True, exist_ok=True)

                current_data = []
                if path.exists() and path.is_file():
                    with open(path, "r") as f:
                        current_data = json.load(f)

                current_data.append(response)

                with open(path, "w") as f:
                    json.dump(current_data, f, indent=4)
            except Exception as e:
                print("An error occurred:", e)
