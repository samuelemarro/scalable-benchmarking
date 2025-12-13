import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from model_config import _slugify, load_registry
from prompt_library import (
    build_critique_prompt,
    build_critique_refine,
    build_critique_self_check,
    load_critique_guidance,
)
from self_improvement import _safe_load_json, self_improve_answers
from utils import query_llm_single

load_dotenv()


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def clean_math(text: str) -> str:
    text = text.replace("\\( ", "$").replace("\\(", "$")
    text = text.replace(" \\)", "$").replace("\\)", "$")
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    return text


def final_question(entry: Dict) -> Optional[str]:
    generations = entry.get("generation_rounds") or []
    if not generations:
        return None
    refinements = generations[-1].get("refinement_rounds") or []
    if not refinements:
        return None
    return refinements[-1].get("question")


def final_answer(entry: Dict) -> Optional[str]:
    attempts = entry.get("attempts") or []
    if not attempts:
        return None
    return attempts[-1].get("answer")


def benchmark_answers(question_model: str, entries: List[Dict]) -> List[Dict]:
    answers: List[Dict] = []
    for entry in entries:
        gen_rounds = entry.get("generation_rounds") or []
        if not gen_rounds:
            answers.append({})
            continue
        refinements = gen_rounds[-1].get("refinement_rounds") or []
        if not refinements:
            answers.append({})
            continue
        last_ref = refinements[-1]
        answers.append(
            {
                "answer_model": question_model,
                "answer": last_ref.get("answer"),
                "status": entry.get("status"),
                "attempts": refinements,
            }
        )
    return answers


def extract_structured_critique(text: Optional[str]) -> Tuple[str, str]:
    """
    Parse a critique JSON (if available) and return verdict and notes (single string).
    Falls back to marking verdict unknown and capturing the raw text as notes.
    """
    verdict = "unknown"
    notes: str = ""
    if not text:
        return verdict, notes

    parsed = _safe_load_json(text)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("verdict"), str):
            verdict = parsed["verdict"]
        raw_notes = parsed.get("notes")
        if isinstance(raw_notes, list):
            notes = "; ".join(str(n) for n in raw_notes)
        elif raw_notes is not None:
            notes = str(raw_notes)

    if not notes and text.strip():
        notes = text.strip()

    return verdict, notes


def pick_answer_record(answer_store: List[Dict], idx: int) -> Optional[Dict]:
    if idx >= len(answer_store):
        return None
    return answer_store[idx]


def prepare_pairs(
    mode: str,
    question_model: str,
    answer_models: List[str],
    answers_root: Path,
    limit: Optional[int],
    benchmark_entries: List[Dict],
) -> List[Tuple[int, str, str]]:
    pairs = []
    q_slug = _slugify(question_model)
    owner_answer_path = answers_root / q_slug / f"{q_slug}.json"
    owner_answers = load_json(owner_answer_path, [])
    if not owner_answers:
        owner_answers = benchmark_answers(question_model, benchmark_entries)

    for answer_model in answer_models:
        a_slug = _slugify(answer_model)
        answer_path = answers_root / q_slug / f"{a_slug}.json"
        answer_records = load_json(answer_path, [])
        if answer_model == question_model and not answer_records:
            answer_records = owner_answers

        if mode == "contradictor":
            # Critics review the question owner's self-answers
            if answer_model == question_model:
                continue
            for idx, owner_entry in enumerate(owner_answers):
                if limit is not None and len(pairs) >= limit:
                    break
                if not owner_entry:
                    continue
                pairs.append((idx, answer_model, question_model))
            continue

        for idx, _ in enumerate(answer_records):
            if limit is not None and len(pairs) >= limit:
                break
            owner_entry = pick_answer_record(owner_answers, idx)
            target_entry = pick_answer_record(answer_records, idx)
            if not owner_entry or not target_entry:
                continue
            if mode == "evaluator":
                if answer_model == question_model:
                    continue
                critic = question_model
                pairs.append((idx, critic, answer_model))
            elif mode in {"all", "custom"}:
                critic = answer_model
                pairs.append((idx, critic, target_entry.get("answer_model", answer_model)))
    return pairs


def generate_single_critique(
    critic_model: str,
    question: str,
    author: str,
    answer: str,
    guidance: str,
    max_rounds: int,
    self_improve: bool,
    disable_batch: bool,
    temperature: float,
    reasoning: Optional[str],
):
    prompt = build_critique_prompt(question, author, answer, guidance)
    if not self_improve:
        critique = query_llm_single(critic_model, prompt, temperature=temperature, reasoning=reasoning)
        critique = clean_math(critique)
        verdict, notes = extract_structured_critique(critique)
        return [
            {"round": 1, "raw_critique": critique, "verdict": verdict, "notes": notes, "evaluation": None},
        ]

    initial = query_llm_single(critic_model, prompt, temperature=temperature, reasoning=reasoning)
    initial = clean_math(initial)

    def eval_prompt(q: str, crit: str, idx: int):
        return build_critique_self_check(q, answer, crit, guidance)

    def refine_prompt(q: str, crit: str, fb: str):
        return build_critique_refine(q, answer, crit, fb)

    results = self_improve_answers(
        critic_model,
        [question],
        [initial],
        eval_prompt,
        refine_prompt,
        max_rounds=max_rounds,
        disable_batch=disable_batch,
        temperature=temperature,
        reasoning=reasoning,
    )[0]

    enriched_attempts = []
    for att in results.attempts:
        verdict, notes = extract_structured_critique(att.answer)
        enriched_attempts.append(
            {
                "round": att.round,
                "raw_critique": att.answer,
                "verdict": verdict,
                "notes": notes,
                "evaluation": att.evaluation,
            }
        )

    return enriched_attempts


def main():
    parser = argparse.ArgumentParser(description="Generate critiques for answers.")
    parser.add_argument("--mode", choices=["contradictor", "evaluator", "all", "custom"], default="contradictor")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--output-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--self-improve", action="store_true", help="Enable critique self-improvement.")
    parser.add_argument("--disable-batch", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of critiques per pair.")
    parser.add_argument("--custom-map", type=Path, help="JSON for custom mode: list of {question_owner, answerer, critic}.")
    parser.add_argument("--models", nargs="*", help="Subset of models to involve (default: all critique-capable).")
    args = parser.parse_args()

    registry = load_registry(str(args.config))
    guidance = load_critique_guidance()

    critic_specs = registry.pick(args.models) if args.models else registry.by_role("critique")

    custom_pairs: List[Tuple[str, str, str]] = []
    if args.mode == "custom":
        if not args.custom_map:
            raise ValueError("Custom mode requires --custom-map")
        payload = load_json(args.custom_map, [])
        for item in payload:
            custom_pairs.append((item["question_owner"], item["answerer"], item["critic"]))

    tasks = []
    with ThreadPoolExecutor(max_workers=max(4, len(critic_specs))) as pool:
        for bench_path in args.benchmark_dir.glob("*.json"):
            q_slug = bench_path.stem
            question_model = next((spec.name for spec in registry.models.values() if spec.slug == q_slug), None)
            if not question_model:
                continue
            benchmark_entries = load_json(bench_path, [])
            answer_models = [spec.name for spec in registry.models.values() if "answer" in spec.roles]

            pairs: List[Tuple[int, str, str]] = []
            if args.mode == "custom":
                for q_owner, answerer, critic in custom_pairs:
                    if q_owner != question_model:
                        continue
                    a_slug = _slugify(answerer)
                    answer_path = args.answers_dir / q_slug / f"{a_slug}.json"
                    if not answer_path.exists():
                        continue
                    answer_records = load_json(answer_path, [])
                    for idx, rec in enumerate(answer_records):
                        if args.limit is not None and len(pairs) >= args.limit:
                            break
                        pairs.append((idx, critic, answerer))
            else:
                pairs = prepare_pairs(args.mode, question_model, answer_models, args.answers_dir, args.limit, benchmark_entries)

            if not pairs:
                continue

            for idx, critic_model, answer_author in pairs:
                critic_spec = registry.models.get(critic_model)
                if not critic_spec or "critique" not in critic_spec.roles:
                    continue
                a_slug = _slugify(answer_author)
                critic_slug = critic_spec.slug
                answers_path = args.answers_dir / q_slug / f"{a_slug}.json"
                answer_records = load_json(answers_path, [])
                if not answer_records and answer_author == question_model:
                    answer_records = benchmark_answers(question_model, benchmark_entries)
                if idx >= len(answer_records):
                    continue
                answer_entry = answer_records[idx]
                question_entry = benchmark_entries[idx]
                question_text = final_question(question_entry)
                if not question_text:
                    continue
                output_path = args.output_dir / args.mode / q_slug / f"{critic_slug}__{a_slug}.json"
                existing = load_json(output_path, [])
                if len(existing) <= idx:
                    existing.extend([{} for _ in range(idx - len(existing) + 1)])

                if existing[idx].get("status") == "succeeded":
                    continue

                def task(
                    qe=question_entry,
                    ae=answer_entry,
                    cs=critic_spec,
                    ans_author=answer_author,
                    out=output_path,
                    record_idx=idx,
                    q_text=question_text,
                    q_author=question_model,
                ):
                    answer_text = final_answer(ae) or ""
                    attempts = generate_single_critique(
                        cs.name,
                        q_text,
                        ans_author,
                        answer_text,
                        guidance,
                        args.max_rounds,
                        args.self_improve,
                        args.disable_batch,
                        cs.temperature,
                        cs.reasoning,
                    )
                    updated = {
                        "question": q_text,
                        "run_id": qe.get("run_id"),
                        "topic_slug": qe.get("topic_slug"),
                        "question_author": q_author,
                        "critic": cs.name,
                        "answer_author": ans_author,
                        "status": "succeeded",
                        "attempts": attempts,
                    }
                    records = load_json(out, [])
                    if len(records) <= record_idx:
                        records.extend([{} for _ in range(record_idx - len(records) + 1)])
                    records[record_idx] = updated
                    save_json(out, records)
                    return cs.name, record_idx

                tasks.append(pool.submit(task))

        for fut in as_completed(tasks):
            try:
                name, idx = fut.result()
                print(f"Critique done by {name} for question #{idx}")
            except Exception as exc:
                print(f"Critique task failed: {exc}")


if __name__ == "__main__":
    main()
