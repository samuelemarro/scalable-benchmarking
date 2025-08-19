import os
import requests
import json

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from utils import query_llm_single

PROMPT = """
Generate the hardest Math question you also have an answer for. It should be hard for another LLM to answer it, but the question must be well-posed and measure Math skills (e.g. "guess what number I'm thinking of" and "find the prime factors of this huge number" do not meaningfully measure Math skills). \
Structure your answer as the [QUESTION] tag, then a problem statement, then a [ANSWER] tag, then the answer. Don't add any other comments besides the problem statement and the answer. Use GitHub Markdown syntax for the question statement and answer (with $ for inline Math and $$ for block math). Do not use a code block for the output
"""

if __name__ == "__main__":
    model = "gpt-4o"

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

        print(response)

        # Serialize the response as a MD file
        with open("response.md", "w") as f:
            f.write(f"Question:\n{response['question']}\n\nAnswer:\n{response['answer']}\n")

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
