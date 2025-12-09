import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

from utils import query_llm_batch

load_dotenv()

# Models, stems, and their temperature/reasoning settings (mirrors generate_answers.py)
MODEL_CONFIGS = [
    ("openai/gpt-5-2025-08-07", "openai-gpt-5-2025-08-07", "high", 1.0), # TODO: Why does it need temperature 1.0?
    ("anthropic/claude-opus-4.1", "anthropic-claude-opus-4.1", "high", 1.0),
    ("google/gemini-2.5-pro", "google-gemini-2.5-pro", "high", 0.0),
    ("openai/gpt-4o-2024-08-06", "openai-gpt-4o-2024-08-06", None, 0.0),
    ("openai/gpt-3.5-turbo", "openai-gpt-3.5-turbo", None, 0.0),
    ("meta-llama/llama-4-maverick", "meta-llama-llama-4-maverick", None, 0.0),
    ("microsoft/phi-4-reasoning-plus", "microsoft-phi-4-reasoning-plus", None, 0.0),
]

SYSTEM_PROMPT = (
    "You are a meticulous math checker. "
    "You will be given a math problem and an answer. "
    "Identify whether there are mistakes in the solution. "
    "Return a compact JSON object with two fields: "
    "`mistakes` (boolean) and `summary` (string with a short rationale). "
    "Do not include any other text."
)


def build_messages(entries: List[Dict[str, Any]]) -> List[str]:
    """
    Build user messages for batch submission.
    Each message includes the problem and the model's previous answer.
    """
    user_messages = []
    for idx, item in enumerate(entries):
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        user_msg = (
            f"Problem #{idx}:\n"
            f"{question}\n\n"
            f"Proposed solution:\n{answer}\n\n"
            "Output JSON only."
        )
        user_messages.append(user_msg)
    return user_messages


def evaluate_model(model: str, benchmark_path: Path, output_path: Path, reasoning: str, temperature: float) -> None:
    with open(benchmark_path, "r") as f:
        entries = json.load(f)

    messages = build_messages(entries)
    responses = query_llm_batch(
        model=model,
        messages_list=messages,
        prompt=SYSTEM_PROMPT,
        temperature=temperature,
        reasoning=reasoning,
        response_format={"type": "json_object"},
    )

    results = []
    for entry, response in zip(entries, responses):
        parsed = {"mistakes": None, "summary": "Unparsed response", "raw": response}
        try:
            data = json.loads(response)
            parsed["mistakes"] = data.get("mistakes")
            parsed["summary"] = data.get("summary", "")
        except Exception:
            # Keep fallback with raw text for debugging
            pass

        results.append(
            {
                "question": entry.get("question"),
                "answer": entry.get("answer"),
                "evaluation": parsed,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    benchmarks_dir = Path("benchmarks")
    output_dir = Path("self-evaluations")

    for model, stem, reasoning, temperature in MODEL_CONFIGS:
        benchmark_path = benchmarks_dir / f"{stem}.json"
        if not benchmark_path.exists():
            print(f"Skipping {model}: missing {benchmark_path}")
            continue

        output_path = output_dir / f"{stem}.json"
        print(f"Evaluating {model} -> {output_path}")
        evaluate_model(model, benchmark_path, output_path, reasoning=reasoning, temperature=temperature)


if __name__ == "__main__":
    main()
