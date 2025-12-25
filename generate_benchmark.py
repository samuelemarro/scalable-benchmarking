import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from model_config import _slugify, load_registry
from prompt_library import (
    build_question_prompt,
    build_refine_prompt,
    build_self_check_prompt,
    load_answer_guidance,
    load_question_guidance,
)
from self_improvement import self_improve_answers
from model_api import query_llm_batch, query_llm_single
from utils import clean_math

load_dotenv()


def load_topic_info(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text())
    info = {}
    for item in data:
        slug = item.get("slug")
        if slug:
            info[slug] = item
    return info


def load_runs(path: Path, topic_info: Dict[str, Dict]) -> List[Dict]:
    data = json.loads(path.read_text())
    runs = []
    for run_id, payload in data.items():
        topic_slug = payload.get("topic")
        if topic_slug not in topic_info:
            raise ValueError(f"Invalid topic slug '{topic_slug}' in run {run_id}. Valid topics: {list(topic_info.keys())}")
        topic_name = topic_info[topic_slug]["name"]
        runs.append({"run_id": str(run_id), "topic_slug": topic_slug, "topic_name": topic_name})
    return runs


def load_existing(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r") as f:
        return json.load(f)


def save_existing(path: Path, data: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def parse_question_answer(text: str) -> Tuple[str, str]:
    if "[QUESTION]" in text and "[ANSWER]" in text:
        question, answer = text.split("[ANSWER]", 1)
        question = question.replace("[QUESTION]", "").strip()
        answer = answer.strip()
        if not question:
            raise ValueError("Parsed question is empty")
        return question, answer
    raise ValueError("Response missing required [QUESTION] and [ANSWER] tags")




def upsert(entries: List[Dict], run_id: str, topic_slug: str, topic_name: str) -> Dict:
    for entry in entries:
        if entry.get("run_id") == run_id:
            return entry
    entry = {
        "run_id": run_id,
        "topic_slug": topic_slug,
        "topic_name": topic_name,
        "status": "pending",
        "generation_rounds": [],
    }
    entries.append(entry)
    return entry


def generate_questions(model: str, runs: List[Dict], prev_map: Dict[str, Dict], disable_batch: bool, guidance: str, temperature: float, reasoning: str):
    prompts = []
    for run in runs:
        run_id = run["run_id"]
        topic_name = run["topic_name"]
        previous_attempts: List[str] = []
        prev_entry = prev_map.get(run_id, {})
        if prev_entry.get("status") == "failed":
            for gen_round in prev_entry.get("generation_rounds", []):
                refinements = gen_round.get("refinement_rounds", [])
                if refinements:
                    question = refinements[-1].get("question")
                    if question:
                        previous_attempts.append(question)
        prompts.append(build_question_prompt(topic_name, guidance, previous_attempts if previous_attempts else None))

    if len(prompts) == 1 or disable_batch:
        return [
            query_llm_single(model, prompt, temperature=temperature, reasoning=reasoning)
            for prompt in prompts
        ]
    return query_llm_batch(model, prompts, temperature=temperature, reasoning=reasoning)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark questions with self-critique.")
    parser.add_argument("--runs-file", type=Path, required=True, help="JSON mapping run IDs to topic slugs.")
    parser.add_argument("--topic-info-file", type=Path, default=Path("configs/topic_info.json"), help="JSON with topic metadata.")
    parser.add_argument("--models", nargs="*", help="Subset of model names to run.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"), help="Model registry JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks"), help="Where to store benchmark files.")
    parser.add_argument("--max-rounds", type=int, default=3, help="Self-critique rounds.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of topics (for testing).")
    parser.add_argument("--disable-batch", action="store_true", help="Disable batching even when available.")
    parser.add_argument("--force-rerun-failures", action="store_true", help="Retry topics marked as failed.")
    args = parser.parse_args()

    registry = load_registry(str(args.config))
    question_guidance = load_question_guidance()
    answer_guidance = load_answer_guidance()
    topic_info = load_topic_info(args.topic_info_file)
    runs = load_runs(args.runs_file, topic_info)
    if args.limit is not None:
        runs = runs[: args.limit]

    models = registry.pick(args.models) if args.models else registry.by_role("benchmark")

    def process_model(model_spec):
        slug = _slugify(model_spec.name)
        output_path = args.output_dir / f"{slug}.json"
        current_entries = load_existing(output_path)

        run_id_to_entry = {entry.get("run_id"): entry for entry in current_entries}
        pending_runs = []

        for run in runs:
            entry = run_id_to_entry.get(run["run_id"])
            if entry and entry.get("status") == "succeeded":
                continue
            if entry and entry.get("status") == "failed" and not args.force_rerun_failures:
                continue
            pending_runs.append(run)

        if not pending_runs:
            return f"No pending topics for {model_spec.name}"

        raw_outputs = generate_questions(
            model_spec.name,
            pending_runs,
            run_id_to_entry,
            args.disable_batch,
            question_guidance,
            model_spec.temperature,
            model_spec.reasoning,
        )

        questions = []
        answers = []
        for run, raw in zip(pending_runs, raw_outputs):
            q, a = parse_question_answer(raw)
            q = clean_math(q)
            a = clean_math(a)
            questions.append(q)
            answers.append(a)

        eval_prompts = lambda q, a, idx: build_self_check_prompt(q, a, answer_guidance)
        refine_prompts = lambda q, a, fb: build_refine_prompt(q, a, fb, answer_guidance)

        improvements = self_improve_answers(
            model_spec.name,
            questions,
            answers,
            eval_prompts,
            refine_prompts,
            max_rounds=args.max_rounds,
            disable_batch=args.disable_batch,
            temperature=model_spec.temperature,
            reasoning=model_spec.reasoning,
        )

        for run, question, result in zip(pending_runs, questions, improvements):
            entry = upsert(current_entries, run["run_id"], run["topic_slug"], run["topic_name"])
            attempt_records = [
                {
                    "round": att.round,
                    "question": question,
                    "answer": att.answer,
                    "evaluation": att.evaluation,
                }
                for att in result.attempts
            ]

            entry["generation_rounds"].append(
                {
                    "refinement_rounds": attempt_records,
                    "status": result.status,
                }
            )

            entry["status"] = result.status

        save_existing(output_path, current_entries)
        return f"Wrote {output_path}"

    with ThreadPoolExecutor(max_workers=max(4, len(models))) as pool:
        futures = [pool.submit(process_model, spec) for spec in models]
        for fut in as_completed(futures):
            try:
                msg = fut.result()
                print(msg)
            except Exception as exc:
                print(f"Generation task failed: {exc}")


if __name__ == "__main__":
    main()
