import argparse
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from model_api import query_llm_batch, query_llm_single
from model_config import ModelSpec, load_registry
from prompt_library import (
    load_answer_guidance,
    load_critique_guidance,
    load_judgment_illposed_guidance,
    load_judgment_critique_guidance,
    load_question_guidance,
)
from utils import safe_load_json, setup_logging
from constants import (
    CRITIQUE_VERDICT_CORRECT,
    JUDGE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
    VALID_CRITIQUE_DEBATE_VERDICTS,
    VALID_ILLPOSED_DEBATE_VERDICTS,
)
from data_models import (
    AnswerEntry,
    AutomatedEvaluation,
    CritiqueEntry,
    DebateMessage,
    EvaluationFile,
    JudgingTask,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_evaluation_entries,
    save_evaluation_entries,
)

logger = logging.getLogger(__name__)

load_dotenv()


def load_decisions(path: Path) -> Dict[str, AutomatedEvaluation]:
    payload = load_evaluation_entries(path)
    return {entry.id: entry for entry in payload.decisions if entry.id}


def save_decisions(path: Path, decisions: Dict[str, AutomatedEvaluation]):
    save_evaluation_entries(path, EvaluationFile(decisions=list(decisions.values())))


def final_answer(entry: AnswerEntry) -> Optional[str]:
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].answer




def build_redactions(registry, alice_model: str, bob_model: str) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    redactions = []
    speaker_map = {"Alice": "Alice", "Bob": "Bob"}
    for name in registry.candidate_model_names(alice_model):
        redactions.append((name, "Alice"))
        speaker_map[name] = "Alice"
    for name in registry.candidate_model_names(bob_model):
        redactions.append((name, "Bob"))
        speaker_map[name] = "Bob"
    return redactions, speaker_map


def redact_text(text: str, redactions: Iterable[Tuple[str, str]]) -> str:
    if not text:
        return ""
    redacted = text
    for needle, replacement in redactions:
        if not needle:
            continue
        redacted = re.sub(re.escape(needle), replacement, redacted, flags=re.IGNORECASE)
    return redacted


def format_debate(
    history: Optional[List[DebateMessage]],
    redactions: Iterable[Tuple[str, str]],
    speaker_map: Dict[str, str],
) -> str:
    if not history:
        return "(No debate transcript available.)"
    lines = []
    for msg in history:
        speaker = speaker_map.get(msg.speaker, msg.speaker)
        message = redact_text(msg.message, redactions)
        if msg.round is not None:
            lines.append(f"- {speaker} (round {msg.round}): {message}")
        else:
            lines.append(f"- {speaker}: {message}")
    return "\n".join(lines)


def gather_illposed_tasks(
    debates_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    registry,
    allow_no_debate: bool = False,
) -> List[JudgingTask]:
    tasks = []
    for debate_file in debates_dir.glob("illposed/*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
        fallback_answers = benchmark_answers_from_entries(
            q_slug,
            benchmark_entries,
        )
        debates = load_debate_entries(debate_file)
        answer_path = answers_dir / q_slug / f"{a_slug}.json"
        if a_slug == q_slug:
            if answer_path.exists():
                raise RuntimeError(
                    f"Self-answer file exists for {q_slug}. Remove {answer_path} to use benchmark answers."
                )
            answers = fallback_answers
        else:
            answers = load_answer_entries(answer_path)
        max_len = max(len(debates), len(answers))
        for idx in range(max_len):
            debate = debates[idx] if idx < len(debates) else None
            answer_record = answers[idx] if idx < len(answers) else None
            status = answer_record.status if answer_record else None
            if status and status != STATUS_ILL_POSED:
                continue
            question = (debate.question if debate else "") or (answer_record.question if answer_record else "")
            answer_text = final_answer(answer_record) if answer_record else ""
            if not answer_text:
                fallback_record = fallback_answers[idx] if idx < len(fallback_answers) else None
                answer_text = final_answer(fallback_record) if fallback_record else ""
                if not question and fallback_record:
                    question = fallback_record.question
            history = debate.history if debate else []
            if not question and not answer_text and not history:
                continue
            if not history and not allow_no_debate:
                logger.warning(f"Skipping illposed/{q_slug}/{a_slug}/{idx}: no debate history (use --allow-no-debate to override)")
                continue
            alice_model = registry.resolve_model_name((debate.alice_model if debate else None) or a_slug)
            bob_model = registry.resolve_model_name((debate.bob_model if debate else None) or q_slug)
            tasks.append(
                JudgingTask(
                    id=f"illposed/{q_slug}/{a_slug}/{idx}",
                    type="illposed",
                    question=question,
                    answer=answer_text,
                    debate_history=history,
                    question_model=q_slug,
                    answer_model=a_slug,
                    alice_model=alice_model,
                    bob_model=bob_model,
                    run_id=(debate.run_id if debate else None) or (answer_record.run_id if answer_record else None),
                    topic_slug=(debate.topic_slug if debate else None) or (answer_record.topic_slug if answer_record else None),
                )
            )
    return tasks


def last_critique_text(crit_entry: CritiqueEntry) -> str:
    attempts = crit_entry.attempts or []
    if not attempts:
        return ""
    last = attempts[-1]
    return last.raw_critique or str(last.notes or "")


def gather_critique_tasks(
    debates_dir: Path,
    critiques_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    registry,
    allow_no_debate: bool = False,
) -> List[JudgingTask]:
    tasks = []
    for mode_dir in critiques_dir.glob("*"):
        mode = mode_dir.name
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
            fallback_answers = benchmark_answers_from_entries(
                q_slug,
                benchmark_entries,
            )
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                critiques = load_critique_entries(crit_file)
                answer_path = answers_dir / q_slug / f"{answer_slug}.json"
                if answer_slug == q_slug:
                    if answer_path.exists():
                        raise RuntimeError(
                            f"Self-answer file exists for {q_slug}. Remove {answer_path} to use benchmark answers."
                        )
                    answers = fallback_answers
                else:
                    answers = load_answer_entries(answer_path)
                debates = load_debate_entries(
                    debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                )
                for idx, crit_entry in enumerate(critiques):
                    if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                        continue
                    attempts = crit_entry.attempts or []
                    last_attempt = attempts[-1] if attempts else None
                    if last_attempt and last_attempt.verdict == CRITIQUE_VERDICT_CORRECT:
                        continue
                    debate = debates[idx] if idx < len(debates) else None
                    question = (debate.question if debate else "") or crit_entry.question
                    answer_record = answers[idx] if idx < len(answers) else None
                    answer_text = final_answer(answer_record) if answer_record else ""
                    if not answer_text:
                        fallback_record = fallback_answers[idx] if idx < len(fallback_answers) else None
                        answer_text = final_answer(fallback_record) if fallback_record else ""
                        if not question and fallback_record:
                            question = fallback_record.question
                    critique_text = last_critique_text(crit_entry)
                    history = debate.history if debate else []
                    if not question and not critique_text and not history:
                        continue
                    if not history and not allow_no_debate:
                        logger.warning(f"Skipping critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}: no debate history (use --allow-no-debate to override)")
                        continue
                    alice_model = registry.resolve_model_name((debate.alice_model if debate else None) or critic_slug)
                    bob_model = registry.resolve_model_name((debate.bob_model if debate else None) or answer_slug)
                    tasks.append(
                        JudgingTask(
                            id=f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}",
                            type="critique",
                            mode=mode,
                            question=question,
                            answer=answer_text,
                            critique=critique_text,
                            debate_history=history,
                            question_model=q_slug,
                            answer_model=answer_slug,
                            critic_model=critic_slug,
                            alice_model=alice_model,
                            bob_model=bob_model,
                            run_id=(debate.run_id if debate else None) or crit_entry.run_id,
                            topic_slug=(debate.topic_slug if debate else None) or crit_entry.topic_slug,
                        )
                    )
    return tasks


def build_illposed_prompt(task: JudgingTask, guidance_q: str, guidance_j: str, registry) -> str:
    redactions, speaker_map = build_redactions(registry, task.alice_model or "", task.bob_model or "")
    question = redact_text(task.question or "", redactions)
    answer = redact_text(task.answer or "", redactions)
    debate = format_debate(task.debate_history, redactions, speaker_map)
    return (
        "You are a neutral judge. Your task: Determine if ALICE'S CLAIM is valid.\n"
        "Alice claims the question is ill-posed. Bob defends that it is well-posed.\n"
        "The debate helps you assess the claim. If Bob concedes or acknowledges the issue, this CONFIRMS Alice was right.\n"
        "Use the guidance below and ignore any party identity cues.\n\n"
        "Judgment guidance:\n"
        f"{guidance_j}\n\n"
        "Question quality guidance:\n"
        f"{guidance_q}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer (context only):\n"
        f"{answer}\n\n"
        "Debate transcript:\n"
        f"{debate}\n\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        '  "verdict": "claimant_wins" | "defender_wins_incorrect" | "wrong_problem" | "mixed" | "unknown",\n'
        '  "confidence": 1-5,\n'
        '  "reasoning": "<concise justification>"\n'
        "}\n\n"
        "Verdict meanings:\n"
        '- "claimant_wins": Alice\'s claim is correct - the question is ill-posed (use this even if Bob concedes)\n'
        '- "defender_wins_incorrect": Alice\'s claim is incorrect - Bob shows the question is well-posed and Alice misidentified the issue\n'
        '- "wrong_problem": The question has issues, but Alice\'s specific diagnosis is incorrect\n'
        '- "mixed": Alice makes multiple claims, some correct and some incorrect\n'
        '- "unknown": Cannot determine if Alice\'s claim is valid with confidence\n\n'
        "Confidence scale: 1=very uncertain, 2=uncertain, 3=moderate, 4=confident, 5=very confident\n"
    )


def build_critique_prompt(task: JudgingTask, guidance_a: str, guidance_c: str, guidance_j: str, registry) -> str:
    redactions, speaker_map = build_redactions(registry, task.alice_model or "", task.bob_model or "")
    question = redact_text(task.question or "", redactions)
    answer = redact_text(task.answer or "", redactions)
    critique = redact_text(task.critique or "", redactions)
    debate = format_debate(task.debate_history, redactions, speaker_map)
    return (
        "You are a neutral judge. Your task: Determine if ALICE'S CRITIQUE is correct.\n"
        "Alice claims Bob's answer has errors/issues. Bob defends the answer.\n"
        "The debate helps you assess the critique. If Bob concedes, fixes the issue, or provides missing justification, this CONFIRMS Alice was right.\n"
        "Use the guidance below and ignore any party identity cues.\n\n"
        "Judgment guidance:\n"
        f"{guidance_j}\n\n"
        "Answer quality guidance:\n"
        f"{guidance_a}\n\n"
        "Critique guidance:\n"
        f"{guidance_c}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:\n"
        f"{answer}\n\n"
        "Critique:\n"
        f"{critique}\n\n"
        "Debate transcript:\n"
        f"{debate}\n\n"
        "Return ONLY a JSON object with this schema:\n"
        "{\n"
        '  "verdict": "claimant_wins" | "defender_wins_incorrect" | "defender_wins_minor" | "wrong_problem" | "mixed" | "unknown",\n'
        '  "confidence": 1-5,\n'
        '  "reasoning": "<concise justification>"\n'
        "}\n\n"
        "Verdict meanings:\n"
        '- "claimant_wins": Alice\'s critique is correct - the answer has substantive flaws (use this even if Bob concedes/fixes)\n'
        '- "defender_wins_incorrect": Alice\'s critique is incorrect - Bob shows the answer is correct and Alice misidentified a problem\n'
        '- "defender_wins_minor": Alice\'s critique is technically correct but about very minor (stylistic only) issues\n'
        '- "wrong_problem": There are issues with the answer, but Alice\'s specific diagnosis is incorrect\n'
        '- "mixed": Alice makes multiple claims, some correct and some incorrect\n'
        '- "unknown": Cannot determine if Alice\'s critique is valid with confidence\n\n'
        "Confidence scale: 1=very uncertain, 2=uncertain, 3=moderate, 4=confident, 5=very confident\n"
    )


def normalize_illposed_verdict(verdict: Optional[str]) -> str:
    """
    Normalize judge verdicts for ill-posedness debates.
    Only accepts canonical forms: claimant_wins, defender_wins_incorrect, wrong_problem, mixed, unknown
    """
    if not verdict:
        return JUDGE_VERDICT_UNKNOWN
    v = verdict.strip().lower()
    if v in VALID_ILLPOSED_DEBATE_VERDICTS:
        return v
    return JUDGE_VERDICT_UNKNOWN


def normalize_critique_verdict(verdict: Optional[str]) -> str:
    """
    Normalize judge verdicts for critique debates.
    Only accepts canonical forms: claimant_wins, defender_wins_incorrect, defender_wins_minor, wrong_problem, mixed, unknown
    """
    if not verdict:
        return JUDGE_VERDICT_UNKNOWN
    v = verdict.strip().lower()
    if v in VALID_CRITIQUE_DEBATE_VERDICTS:
        return v
    return JUDGE_VERDICT_UNKNOWN


def parse_confidence(raw) -> int:
    """
    Parse confidence. Only accepts integers 1-5.
    Raises ValueError if the value is invalid or missing.
    """
    if raw is None:
        raise ValueError("Confidence field is required")
    try:
        conf = int(raw)
        if 1 <= conf <= 5:
            return conf
        raise ValueError(f"Confidence must be 1-5, got: {conf}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid confidence value: {raw}") from e


def parse_judgment(text: str, task: JudgingTask, judge_slug: str) -> AutomatedEvaluation:
    schema_hint = (
        '{"verdict": "...", "confidence": "...", "reasoning": "..."}'
    )
    parsed = safe_load_json(text or "", schema_hint=schema_hint)
    verdict = None
    confidence = None
    reasoning = None
    if isinstance(parsed, dict):
        verdict = parsed.get("verdict")
        # Catch confidence parsing exceptions and mark judgment as failed
        try:
            confidence = parse_confidence(parsed.get("confidence"))
        except ValueError as e:
            logger.warning(f"Failed to parse confidence for task {task.id}: {e}")
            confidence = None
            # Mark as failed if confidence is required but missing/invalid
            verdict = JUDGE_VERDICT_UNKNOWN
        reasoning = parsed.get("reasoning", None)
    if task.type == "illposed":
        verdict = normalize_illposed_verdict(verdict)
    else:
        verdict = normalize_critique_verdict(verdict)
    status = STATUS_SUCCEEDED if verdict not in {JUDGE_VERDICT_UNKNOWN, "invalid"} else STATUS_FAILED
    return AutomatedEvaluation(
        id=task.id,
        type=task.type,
        mode=task.mode,
        question_model=task.question_model,
        answer_model=task.answer_model,
        critic_model=task.critic_model,
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        status=status,
        raw_response=text,
        run_id=task.run_id,
        topic_slug=task.topic_slug,
        judge_model=judge_slug,
    )


def chunked(items: List[JudgingTask], size: Optional[int]) -> Iterable[List[JudgingTask]]:
    if not size or size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _batched_query(
    model: str,
    prompts: List[str],
    disable_batch: bool,
    temperature: Optional[float],
    reasoning: Optional[str],
) -> List[str]:
    if len(prompts) == 1 or disable_batch:
        return [query_llm_single(model, prompts[0], temperature=temperature, reasoning=reasoning)]
    return query_llm_batch(model, prompts, temperature=temperature, reasoning=reasoning)


def main():
    parser = argparse.ArgumentParser(description="Automated judging of debate claims.")
    parser.add_argument("--mode", choices=["illposed", "critiques", "all"], default="all")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--output-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--disable-batch", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit tasks per judge.")
    parser.add_argument("--models", nargs="*", help="Subset of judge models to use (default: all).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-no-debate", action="store_true", help="Allow judging even when there's no debate history.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    args = parser.parse_args()

    setup_logging(args.log_level)

    registry = load_registry(str(args.config))
    guidance_q = load_question_guidance()
    guidance_a = load_answer_guidance()
    guidance_c = load_critique_guidance()
    guidance_j_illposed = load_judgment_illposed_guidance()
    guidance_j_critique = load_judgment_critique_guidance()

    tasks: List[JudgingTask] = []
    if args.mode in {"illposed", "all"}:
        tasks.extend(gather_illposed_tasks(args.debates_dir, args.answers_dir, args.benchmark_dir, registry, args.allow_no_debate))
    if args.mode in {"critiques", "all"}:
        tasks.extend(
            gather_critique_tasks(
                args.debates_dir,
                args.critiques_dir,
                args.answers_dir,
                args.benchmark_dir,
                registry,
                args.allow_no_debate,
            )
        )

    judges = registry.pick(args.models) if args.models else list(registry.models.values())
    jobs_by_judge: Dict[str, Dict[str, object]] = {spec.name: {"spec": spec, "tasks": []} for spec in judges}

    for task in tasks:
        participants = {task.alice_model, task.bob_model}
        for spec in judges:
            if spec.name in participants:
                continue
            jobs_by_judge[spec.name]["tasks"].append(task)

    def process_judge(spec: ModelSpec, tasks_for_judge: List[JudgingTask]) -> int:
        out_path = args.output_dir / f"{spec.slug}.json"
        decisions = load_decisions(out_path)
        pending = []
        for task in tasks_for_judge:
            if not args.overwrite and task.id in decisions:
                continue
            pending.append(task)
            if args.limit is not None and len(pending) >= args.limit:
                break
        if not pending:
            return 0

        for batch in chunked(pending, args.batch_size):
            prompts = []
            for task in batch:
                if task.type == "illposed":
                    prompt = build_illposed_prompt(task, guidance_q, guidance_j_illposed, registry)
                else:
                    prompt = build_critique_prompt(task, guidance_a, guidance_c, guidance_j_critique, registry)
                prompts.append(prompt)
            try:
                responses = _batched_query(
                    spec.name,
                    prompts,
                    args.disable_batch,
                    spec.temperature,
                    spec.reasoning,
                )
            except Exception as exc:
                logger.error(f"Judge batch failed for {spec.name}: {exc}")
                continue
            for task, response in zip(batch, responses):
                decision = parse_judgment(response, task, spec.slug)
                decisions[task.id] = decision
            save_decisions(out_path, decisions)
            logger.info(f"{spec.pretty}: saved {len(batch)} evaluations")
        return len(pending)

    with ThreadPoolExecutor(max_workers=max(4, len(jobs_by_judge))) as pool:
        futures = []
        for payload in jobs_by_judge.values():
            spec = payload["spec"]
            tasks_for_judge = payload["tasks"]
            if not tasks_for_judge:
                continue
            futures.append(pool.submit(process_judge, spec, tasks_for_judge))
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                logger.error(f"Judge worker failed: {exc}")


if __name__ == "__main__":
    main()
