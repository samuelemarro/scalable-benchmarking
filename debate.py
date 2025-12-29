import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from model_config import _slugify, load_registry
from prompt_library import (
    load_answer_guidance,
    load_critique_guidance,
    load_debate_illposed_guidance,
    load_debate_critique_guidance,
    load_question_guidance,
)
from model_api import query_llm_single
from utils import safe_load_json, clean_math, setup_logging
from constants import CRITIQUE_VERDICT_CORRECT, STATUS_ILL_POSED, STATUS_SUCCEEDED
from data_models import (
    AnswerAttempt,
    AnswerEntry,
    BenchmarkEntry,
    CritiqueAttempt,
    CritiqueEntry,
    DebateEntry,
    DebateMessage,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    save_debate_entries,
)

logger = logging.getLogger(__name__)

load_dotenv()


def final_answer(entry: AnswerEntry) -> Optional[str]:
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].answer


def resolved_critique_text(entry: CritiqueEntry) -> str:
    attempts = entry.attempts or []
    if attempts:
        last = attempts[-1]
        if last.raw_critique:
            return last.raw_critique
        if last.notes:
            return str(last.notes)
    return ""


def final_critique_verdict(entry: CritiqueEntry) -> Optional[str]:
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].verdict


def run_round(
    speaker_model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    reasoning: Optional[str],
) -> Tuple[str, bool]:
    reply = query_llm_single(
        speaker_model,
        user_message,
        prompt=system_prompt + "\n\nIMPORTANT: Respond with JSON containing 'message' (your response text) and 'concede' (boolean, true if you concede/agree, false otherwise).",
        temperature=temperature,
        reasoning=reasoning,
        response_format={"type": "json_object"},
    )
    parsed = safe_load_json(reply)
    if isinstance(parsed, dict) and "message" in parsed:
        return clean_math(parsed["message"]), parsed.get("concede", False)
    # Fallback if JSON parsing fails
    return clean_math(reply), False


def check_concession(concede: bool) -> bool:
    """Check if the response contains a concession via the 'concede' field."""
    return bool(concede)


def illposed_debate(
    defender_model: str,
    claimant_model: str,
    question: str,
    claim_details: Dict,
    rounds: int,
    guidance_q: str,
    guidance_d: str,
    defender_temp: float,
    defender_reason: Optional[str],
    claimant_temp: float,
    claimant_reason: Optional[str],
    allow_concede: bool,
) -> List[DebateMessage]:
    history: List[DebateMessage] = []
    system_prompt_owner = (
        "You are Bob, defending the well-posedness of your question. Use the guidance to rebut the ill-posedness claim."
        f"\nQuestion quality guidance:\n{guidance_q}\n\n"
        f"Debate guidance:\n{guidance_d}\n"
    )
    system_prompt_claimant = (
        "You are Alice, justifying why the question is ill-posed. Stay concise and evidence-based."
        f"\nQuestion quality guidance:\n{guidance_q}\n\n"
        f"Debate guidance:\n{guidance_d}\n"
    )
    last_message = (
        f"Alice argues your question is ill-posed.\nQuestion:\n{question}\n\nClaim:\n{json.dumps(claim_details, indent=2)}\n"
        "Respond with a short defense as Bob."
    )
    for r in range(1, rounds + 1):
        owner_message, owner_concede = run_round(defender_model, system_prompt_owner, last_message, defender_temp, defender_reason)
        history.append(DebateMessage(round=r, speaker="Bob", message=owner_message, concede=owner_concede))
        if allow_concede and check_concession(owner_concede):
            break
        claimant_prompt = (
            f"Bob responded:\n{owner_message}\n\nQuestion:\n{question}\n\n"
            "Restate your ill-posedness reasoning or acknowledge if the defense resolves your concerns as Alice."
        )
        claimant_message, claimant_concede = run_round(claimant_model, system_prompt_claimant, claimant_prompt, claimant_temp, claimant_reason)
        history.append(DebateMessage(round=r, speaker="Alice", message=claimant_message, concede=claimant_concede))
        if allow_concede and check_concession(claimant_concede):
            break
        last_message = (
            f"Alice replied:\n{claimant_message}\n\nQuestion:\n{question}\n"
            "Respond briefly to move the discussion forward as Bob."
        )
    return history


def critique_debate(
    defender_model: str,
    claimant_model: str,
    question: str,
    answer: str,
    critique: str,
    rounds: int,
    guidance_a: str,
    guidance_c: str,
    guidance_d: str,
    author_temp: float,
    author_reason: Optional[str],
    critic_temp: float,
    critic_reason: Optional[str],
    allow_concede: bool,
) -> List[DebateMessage]:
    history: List[DebateMessage] = []
    system_prompt_author = (
        "You are Bob, responding to Alice's critique of your answer. Be concise and correct errors when valid."
        f"\nAnswer quality guidance:\n{guidance_a}\n\n"
        f"Debate guidance:\n{guidance_d}\n"
    )
    system_prompt_critic = (
        "You are Alice, continuing your critique. Acknowledge fixes if they address the issue. Stay factual."
        f"\nCritique quality guidance:\n{guidance_c}\n\n"
        f"Debate guidance:\n{guidance_d}\n"
    )
    last_message = (
        f"Alice raised a critique about your answer.\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nCritique:\n{critique}\n"
        "Respond briefly as Bob."
    )
    for r in range(1, rounds + 1):
        author_message, author_concede = run_round(defender_model, system_prompt_author, last_message, author_temp, author_reason)
        history.append(DebateMessage(round=r, speaker="Bob", message=author_message, concede=author_concede))
        if allow_concede and check_concession(author_concede):
            break
        critic_prompt = (
            f"Bob replied:\n{author_message}\n\nOriginal critique:\n{critique}\n\nQuestion:\n{question}\n"
            "Follow up concisely as Alice."
        )
        critic_message, critic_concede = run_round(claimant_model, system_prompt_critic, critic_prompt, critic_temp, critic_reason)
        history.append(DebateMessage(round=r, speaker="Alice", message=critic_message, concede=critic_concede))
        if allow_concede and check_concession(critic_concede):
            break
        last_message = (
            f"Alice replied:\n{critic_message}\n\nQuestion:\n{question}\nAnswer:\n{answer}\n"
            "Respond briefly as Bob."
        )
    return history


def main():
    parser = argparse.ArgumentParser(description="Facilitate debates on ill-posedness claims or critiques.")
    parser.add_argument("--mode", choices=["ill-posed", "critique"], required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--output-dir", type=Path, default=Path("debates"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-allow-concede", action="store_true", help="Disable early stop on concession.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.rounds < 1:
        parser.error("--rounds must be >= 1")

    registry = load_registry(str(args.config))
    guidance_q = load_question_guidance()
    guidance_a = load_answer_guidance()
    guidance_c = load_critique_guidance()
    guidance_d_illposed = load_debate_illposed_guidance()
    guidance_d_critique = load_debate_critique_guidance()

    debates = 0

    if args.mode == "ill-posed":
        # Count total tasks for progress bar
        total_tasks = 0
        for bench_path in args.benchmark_dir.glob("*.json"):
            q_slug = bench_path.stem
            question_model = next((spec for spec in registry.models.values() if spec.slug == q_slug), None)
            if not question_model:
                continue
            answers_dir = args.answers_dir / q_slug
            for answer_file in answers_dir.glob("*.json"):
                answer_model_slug = answer_file.stem
                answer_model = next((spec for spec in registry.models.values() if spec.slug == answer_model_slug), None)
                if not answer_model:
                    continue
                if answer_model_slug == q_slug:
                    raise RuntimeError(
                        f"Self-answer file exists for {q_slug}. Remove {answer_file} to use benchmark answers."
                    )
                records = load_answer_entries(answer_file)
                for idx, rec in enumerate(records):
                    if args.limit is not None and total_tasks >= args.limit:
                        break
                    if not rec or rec.status != STATUS_ILL_POSED:
                        continue
                    debate_path = args.output_dir / "illposed" / q_slug / f"{answer_model_slug}.json"
                    existing = load_debate_entries(debate_path)
                    if len(existing) > idx and existing[idx]:
                        continue
                    total_tasks += 1
                if args.limit is not None and total_tasks >= args.limit:
                    break
            if args.limit is not None and total_tasks >= args.limit:
                break

        # Process with progress bar
        pbar = tqdm(total=total_tasks, desc="Ill-posed debates")
        for bench_path in args.benchmark_dir.glob("*.json"):
            q_slug = bench_path.stem
            question_model = next((spec for spec in registry.models.values() if spec.slug == q_slug), None)
            if not question_model:
                continue
            answers_dir = args.answers_dir / q_slug
            for answer_file in answers_dir.glob("*.json"):
                answer_model_slug = answer_file.stem
                answer_model = next((spec for spec in registry.models.values() if spec.slug == answer_model_slug), None)
                if not answer_model:
                    continue
                if answer_model_slug == q_slug:
                    raise RuntimeError(
                        f"Self-answer file exists for {q_slug}. Remove {answer_file} to use benchmark answers."
                    )
                records = load_answer_entries(answer_file)

                # Load debate file once per answer_file instead of per record
                debate_path = args.output_dir / "illposed" / q_slug / f"{answer_model_slug}.json"
                existing = load_debate_entries(debate_path)

                for idx, rec in enumerate(records):
                    if args.limit is not None and debates >= args.limit:
                        break
                    if not rec or rec.status != STATUS_ILL_POSED:
                        continue
                    claim = rec.ill_posed_claim or {}

                    if len(existing) > idx and existing[idx]:
                        continue

                    try:
                        history = illposed_debate(
                            question_model.name,
                            answer_model.name,
                            rec.question,
                            claim,
                            args.rounds,
                            guidance_q,
                            guidance_d_illposed,
                            question_model.temperature,
                            question_model.reasoning,
                            answer_model.temperature,
                            answer_model.reasoning,
                            not args.no_allow_concede,
                        )
                        if len(existing) <= idx:
                            existing.extend([None for _ in range(idx - len(existing) + 1)])
                        existing[idx] = DebateEntry(
                            question=rec.question,
                            alice_model=answer_model.slug,
                            bob_model=question_model.slug,
                            claimant=answer_model.slug,
                            run_id=rec.run_id,
                            topic_slug=rec.topic_slug,
                            history=history,
                        )
                        save_debate_entries(debate_path, existing)
                        debates += 1
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Failed to generate ill-posed debate for {q_slug}/{answer_model_slug}/{idx}: {e}")
                        pbar.update(1)
                        continue
                if args.limit is not None and debates >= args.limit:
                    break
            if args.limit is not None and debates >= args.limit:
                break
        pbar.close()
    else:
        # critique debates
        # Count total tasks for progress bar
        total_tasks = 0
        for crit_mode_dir in args.critiques_dir.glob("*"):
            mode = crit_mode_dir.name
            for q_dir in crit_mode_dir.glob("*"):
                q_slug = q_dir.name
                question_model = next((spec for spec in registry.models.values() if spec.slug == q_slug), None)
                if not question_model:
                    continue
                benchmark_entries = load_benchmark_entries(args.benchmark_dir / f"{q_slug}.json")
                for crit_file in q_dir.glob("*.json"):
                    parts = crit_file.stem.split("__")
                    if len(parts) != 2:
                        continue
                    critic_slug, answer_slug = parts
                    critic_model = next((spec for spec in registry.models.values() if spec.slug == critic_slug), None)
                    answer_model = next((spec for spec in registry.models.values() if spec.slug == answer_slug), None)
                    if not critic_model or not answer_model:
                        continue
                    critiques = load_critique_entries(crit_file)
                    answer_path = args.answers_dir / q_slug / f"{answer_slug}.json"
                    if answer_slug == q_slug:
                        if answer_path.exists():
                            raise RuntimeError(
                                f"Self-answer file exists for {q_slug}. Remove {answer_path} to use benchmark answers."
                            )
                        answers = benchmark_answers_from_entries(
                            q_slug,
                            benchmark_entries,
                        )
                    else:
                        answers = load_answer_entries(answer_path)
                    for idx, crit_entry in enumerate(critiques):
                        if args.limit is not None and total_tasks >= args.limit:
                            break
                        if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                            continue
                        if final_critique_verdict(crit_entry) == CRITIQUE_VERDICT_CORRECT:
                            continue
                        if idx >= len(answers):
                            continue
                        debate_path = args.output_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                        existing = load_debate_entries(debate_path)
                        if len(existing) > idx and existing[idx]:
                            continue
                        total_tasks += 1
                    if args.limit is not None and total_tasks >= args.limit:
                        break
                if args.limit is not None and total_tasks >= args.limit:
                    break
            if args.limit is not None and total_tasks >= args.limit:
                break

        # Process with progress bar
        pbar = tqdm(total=total_tasks, desc="Critique debates")
        for crit_mode_dir in args.critiques_dir.glob("*"):
            mode = crit_mode_dir.name
            for q_dir in crit_mode_dir.glob("*"):
                q_slug = q_dir.name
                question_model = next((spec for spec in registry.models.values() if spec.slug == q_slug), None)
                if not question_model:
                    continue
                benchmark_entries = load_benchmark_entries(args.benchmark_dir / f"{q_slug}.json")
                for crit_file in q_dir.glob("*.json"):
                    parts = crit_file.stem.split("__")
                    if len(parts) != 2:
                        continue
                    critic_slug, answer_slug = parts
                    critic_model = next((spec for spec in registry.models.values() if spec.slug == critic_slug), None)
                    answer_model = next((spec for spec in registry.models.values() if spec.slug == answer_slug), None)
                    if not critic_model or not answer_model:
                        continue
                    critiques = load_critique_entries(crit_file)
                    answer_path = args.answers_dir / q_slug / f"{answer_slug}.json"
                    if answer_slug == q_slug:
                        if answer_path.exists():
                            raise RuntimeError(
                                f"Self-answer file exists for {q_slug}. Remove {answer_path} to use benchmark answers."
                            )
                        answers = benchmark_answers_from_entries(
                            q_slug,
                            benchmark_entries,
                        )
                    else:
                        answers = load_answer_entries(answer_path)

                    # Load debate file once per critique file instead of per record
                    debate_path = args.output_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                    existing = load_debate_entries(debate_path)

                    for idx, crit_entry in enumerate(critiques):
                        if args.limit is not None and debates >= args.limit:
                            break
                        if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                            continue
                        if final_critique_verdict(crit_entry) == CRITIQUE_VERDICT_CORRECT:
                            continue
                        if idx >= len(answers):
                            continue
                        answer_entry = answers[idx]

                        if len(existing) > idx and existing[idx]:
                            continue
                        if not answer_entry:
                            continue
                        answer_text = final_answer(answer_entry) or ""
                        critique_text = resolved_critique_text(crit_entry)

                        try:
                            history = critique_debate(
                                answer_model.name,
                                critic_model.name,
                                crit_entry.question,
                                answer_text,
                                critique_text,
                                args.rounds,
                                guidance_a,
                                guidance_c,
                                guidance_d_critique,
                                answer_model.temperature,
                                answer_model.reasoning,
                                critic_model.temperature,
                                critic_model.reasoning,
                                not args.no_allow_concede,
                            )
                            if len(existing) <= idx:
                                existing.extend([None for _ in range(idx - len(existing) + 1)])
                            existing[idx] = DebateEntry(
                                question=crit_entry.question,
                                alice_model=critic_model.slug,
                                bob_model=answer_model.slug,
                                run_id=crit_entry.run_id,
                                topic_slug=crit_entry.topic_slug,
                                answer_author=answer_model.slug,
                                critic=critic_model.slug,
                                history=history,
                            )
                            save_debate_entries(debate_path, existing)
                            debates += 1
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Failed to generate critique debate for {mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}: {e}")
                            pbar.update(1)
                            continue
                    if args.limit is not None and debates >= args.limit:
                        break
                if args.limit is not None and debates >= args.limit:
                    break
            if args.limit is not None and debates >= args.limit:
                break
        pbar.close()

    logger.info(f"Generated {debates} debate transcripts.")


if __name__ == "__main__":
    main()
