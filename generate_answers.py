import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

from model_config import _slugify, load_registry
from prompt_library import (
    build_answer_prompt,
    build_refine_prompt,
    build_self_check_prompt,
    load_answer_guidance,
    load_self_critique_guidance,
)
from self_improvement import self_improve_answers
from model_api import query_llm_batch, query_llm_single
from constants import STATUS_FAILED, STATUS_ILL_POSED, STATUS_SUCCEEDED
from data_models import (
    AnswerAttempt,
    AnswerEntry,
    BenchmarkEntry,
    load_answer_entries,
    load_benchmark_entries,
    save_answer_entries,
)
from utils import clean_math, setup_logging

logger = logging.getLogger(__name__)

load_dotenv()


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def build_override_note(overrides: Dict, q_slug: str, a_slug: str, idx: int) -> Optional[str]:
    key = f"{q_slug}/{a_slug}"
    reason = overrides.get("overrides", {}).get(key, {}).get(str(idx))
    if reason:
        return f"Override present: do not mark this question as ill-posed. Reason: {reason}."
    return None


def final_question(entry) -> Optional[str]:
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question


def prepare_batch(
    question_model: str,
    answer_model: str,
    benchmark_entries: List[Optional[BenchmarkEntry]],
    existing: List[Optional[AnswerEntry]],
    rerun_failures: bool,
    limit: Optional[int],
    allow_self_answering: bool,
) -> List[Tuple[int, Dict, str]]:
    if question_model == answer_model and not allow_self_answering:
        return []
    batch = []
    for idx, entry in enumerate(benchmark_entries):
        if limit is not None and len(batch) >= limit:
            break
        if not entry or entry.status != STATUS_SUCCEEDED:
            status = entry.status if entry else "missing"
            logger.info(f"Skipping question {question_model}-{idx} (status: {status})")
            continue
        question_text = final_question(entry)
        if not question_text:
            continue
        if idx < len(existing):
            prior = existing[idx]
            if prior and prior.status == STATUS_SUCCEEDED:
                continue
            if prior and prior.status in {STATUS_FAILED, STATUS_ILL_POSED} and not rerun_failures:
                continue
        batch.append((idx, entry, question_text))
    return batch


def run_generation(
    question_model: str,
    answer_model: str,
    batch_items: List[Tuple[int, BenchmarkEntry, str]],
    answer_guidance: str,
    self_critique_guidance: str,
    max_rounds: int,
    disable_batch: bool,
    temperature: float,
    reasoning: Optional[str],
    override_payload: Dict,
):
    questions = [item[2] for item in batch_items]
    prompts = [build_answer_prompt(q, answer_guidance) for q in questions]

    q_slug = _slugify(question_model)
    a_slug = _slugify(answer_model)

    if len(prompts) == 1 or disable_batch:
        raw_answers = [
            query_llm_single(answer_model, prompt, temperature=temperature, reasoning=reasoning)
            for prompt in prompts
        ]
    else:
        raw_answers = query_llm_batch(answer_model, prompts, temperature=temperature, reasoning=reasoning)

    cleaned_answers = [clean_math(r) for r in raw_answers]

    def eval_prompt(question: str, answer: str, local_idx: int):
        note = build_override_note(override_payload, q_slug, a_slug, batch_items[local_idx][0])
        base = build_self_check_prompt(question, answer, self_critique_guidance, answer_guidance)
        return base + f"\n\n{note}" if note else base
    refine_prompts = lambda q, a, fb: build_refine_prompt(q, a, fb, answer_guidance)

    results = self_improve_answers(
        answer_model,
        questions,
        cleaned_answers,
        eval_prompt,
        refine_prompts,
        max_rounds=max_rounds,
        disable_batch=disable_batch,
        temperature=temperature,
        reasoning=reasoning,
        raw_initial_answers=raw_answers,
    )

    outputs = []
    for (idx, entry, question_text), improved in zip(batch_items, results):
        attempts = [
            AnswerAttempt(
                round=att.round,
                answer=att.answer,
                raw_answer=att.raw_answer,
                evaluation=att.evaluation,
            )
            for att in improved.attempts
        ]
        ill_posed_claim = None
        if improved.status == STATUS_ILL_POSED:
            if improved.attempts:
                ill_posed_claim = improved.attempts[-1].evaluation
            if ill_posed_claim is None:
                ill_posed_claim = improved.last_feedback
        record = AnswerEntry(
            question_model=q_slug,
            answer_model=a_slug,
            question=question_text,
            run_id=entry.run_id,
            topic_slug=entry.topic_slug,
            ill_posed_claim=ill_posed_claim,
            status=improved.status,
            attempts=attempts,
        )
        outputs.append((idx, record))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Generate answers for benchmark questions with self-critique.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"), help="Model registry.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"), help="Directory with benchmark JSON files.")
    parser.add_argument("--output-dir", type=Path, default=Path("answers"), help="Directory to store answers.")
    parser.add_argument("--models", nargs="*", help="Subset of models to use as answerers.")
    parser.add_argument("--max-rounds", type=int, default=5, help="Self-improvement rounds.")
    parser.add_argument("--disable-batch", action="store_true", help="Disable batching.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions per pair.")
    parser.add_argument("--rerun-failures", action="store_true", help="Retry failed entries.")
    parser.add_argument("--allow-self-answering", action="store_true", help="Allow models to answer their own questions.")
    parser.add_argument("--illposed-overrides", type=Path, default=Path("configs/illposed_overrides.json"), help="Overrides JSON.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    args = parser.parse_args()

    setup_logging(args.log_level)

    registry = load_registry(str(args.config))
    answer_guidance = load_answer_guidance()
    self_critique_guidance = load_self_critique_guidance()
    overrides = load_json(args.illposed_overrides, {"overrides": {}})

    answer_models = registry.pick(args.models) if args.models else registry.by_role("answer")
    benchmark_files = list(args.benchmark_dir.glob("*.json"))

    tasks = []
    with ThreadPoolExecutor(max_workers=max(4, len(answer_models))) as pool:
        for bench_path in benchmark_files:
            q_slug = bench_path.stem
            question_model = next((spec.name for spec in registry.models.values() if _slugify(spec.name) == q_slug), None)
            if not question_model:
                continue
            benchmark_entries = load_benchmark_entries(bench_path)
            if not benchmark_entries:
                logger.info(f"Skipping empty benchmark: {bench_path}")
                continue
            for answer_spec in answer_models:
                if answer_spec.name == question_model and not args.allow_self_answering:
                    continue
                out_path = args.output_dir / q_slug / f"{answer_spec.slug}.json"
                existing = load_answer_entries(out_path)
                batch = prepare_batch(
                    question_model,
                    answer_spec.name,
                    benchmark_entries,
                    existing,
                    args.rerun_failures,
                    args.limit,
                    args.allow_self_answering,
                )
                if not batch:
                    continue

                def run_task(qm=question_model, am=answer_spec, b=batch, b_entries=benchmark_entries, out=out_path):
                    outputs = run_generation(
                        qm,
                        am.name,
                        b,
                        answer_guidance,
                        self_critique_guidance,
                        args.max_rounds,
                        args.disable_batch,
                        am.temperature,
                        am.reasoning,
                        overrides,
                    )
                    existing_records = load_answer_entries(out)
                    # Ensure list length
                    if len(existing_records) < len(b_entries):
                        existing_records.extend([None for _ in range(len(b_entries) - len(existing_records))])
                    for idx, record in outputs:
                        existing_records[idx] = record
                    save_answer_entries(out, existing_records)
                    return qm, am.name, len(outputs)

                tasks.append(pool.submit(run_task))

        for fut in as_completed(tasks):
            try:
                qm, am, count = fut.result()
                logger.info(f"Finished {count} answers for {am} on {qm}")
            except Exception as exc:
                logger.error(f"Task failed: {exc}")


if __name__ == "__main__":
    main()
