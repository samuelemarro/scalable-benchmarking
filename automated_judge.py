import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

from model_api import query_llm_batch, query_llm_single
from model_config import ModelSpec, _slugify, load_registry
from prompt_library import load_answer_guidance, load_critique_guidance, load_question_guidance
from utils import safe_load_json

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


def load_decisions(path: Path) -> Dict[str, Dict]:
    payload = load_json(path, {"decisions": []})
    decisions = {}
    for entry in payload.get("decisions", []):
        entry_id = entry.get("id")
        if entry_id:
            decisions[entry_id] = entry
    return decisions


def save_decisions(path: Path, decisions: Dict[str, Dict]):
    save_json(path, {"decisions": list(decisions.values())})


def final_answer(entry: Dict) -> Optional[str]:
    attempts = entry.get("attempts") or []
    if not attempts:
        return None
    return attempts[-1].get("answer")


def benchmark_answer_for_index(benchmark_dir: Path, q_slug: str, idx: int) -> Optional[str]:
    bench_path = benchmark_dir / f"{q_slug}.json"
    entries = load_json(bench_path, [])
    if idx >= len(entries):
        return None
    entry = entries[idx]
    generations = entry.get("generation_rounds") or []
    if not generations:
        return None
    refinements = generations[-1].get("refinement_rounds") or []
    if not refinements:
        return None
    return refinements[-1].get("answer")




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


def format_debate(history: List[Dict], redactions: Iterable[Tuple[str, str]], speaker_map: Dict[str, str]) -> str:
    if not history:
        return "(No debate transcript available.)"
    lines = []
    for msg in history:
        speaker = msg.get("speaker") or "Speaker"
        speaker = speaker_map.get(speaker, speaker)
        message = redact_text(msg.get("message", ""), redactions)
        round_no = msg.get("round")
        if round_no is not None:
            lines.append(f"- {speaker} (round {round_no}): {message}")
        else:
            lines.append(f"- {speaker}: {message}")
    return "\n".join(lines)


def gather_illposed_tasks(
    debates_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    registry,
) -> List[Dict]:
    tasks = []
    for debate_file in debates_dir.glob("illposed/*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        debates = load_json(debate_file, [])
        answers = load_json(answers_dir / q_slug / f"{a_slug}.json", [])
        max_len = max(len(debates), len(answers))
        for idx in range(max_len):
            debate = debates[idx] if idx < len(debates) else {}
            answer_record = answers[idx] if idx < len(answers) else {}
            status = answer_record.get("status")
            if status and status != "ill-posed":
                continue
            question = debate.get("question") or answer_record.get("question", "")
            answer_text = final_answer(answer_record) or answer_record.get("answer") or ""
            if not answer_text:
                answer_text = benchmark_answer_for_index(benchmark_dir, q_slug, idx) or ""
            history = debate.get("history", [])
            if not question and not answer_text and not history:
                continue
            alice_model = registry.resolve_model_name(debate.get("alice_model") or a_slug)
            bob_model = registry.resolve_model_name(debate.get("bob_model") or q_slug)
            tasks.append(
                {
                    "id": f"illposed/{q_slug}/{a_slug}/{idx}",
                    "type": "illposed",
                    "question": question,
                    "answer": answer_text,
                    "debate": history,
                    "question_model": registry.display_name_for_slug(q_slug),
                    "answer_model": registry.display_name_for_slug(a_slug),
                    "alice_model": alice_model,
                    "bob_model": bob_model,
                    "run_id": debate.get("run_id") or answer_record.get("run_id"),
                    "topic_slug": debate.get("topic_slug") or answer_record.get("topic_slug"),
                }
            )
    return tasks


def last_critique_text(crit_entry: Dict) -> str:
    attempts = crit_entry.get("attempts") or []
    if not attempts:
        return ""
    last = attempts[-1]
    return last.get("raw_critique") or str(last.get("notes") or "")


def gather_critique_tasks(
    debates_dir: Path,
    critiques_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    registry,
) -> List[Dict]:
    tasks = []
    for mode_dir in critiques_dir.glob("*"):
        mode = mode_dir.name
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                critiques = load_json(crit_file, [])
                answers = load_json(answers_dir / q_slug / f"{answer_slug}.json", [])
                debates = load_json(
                    debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json",
                    [],
                )
                for idx, crit_entry in enumerate(critiques):
                    if not crit_entry or crit_entry.get("status") != "succeeded":
                        continue
                    attempts = crit_entry.get("attempts") or []
                    last_attempt = attempts[-1] if attempts else {}
                    if last_attempt.get("verdict") == "correct answer":
                        continue
                    debate = debates[idx] if idx < len(debates) else {}
                    question = debate.get("question") or crit_entry.get("question", "")
                    answer_record = answers[idx] if idx < len(answers) else {}
                    answer_text = final_answer(answer_record) or answer_record.get("answer") or ""
                    if not answer_text:
                        answer_text = benchmark_answer_for_index(benchmark_dir, q_slug, idx) or ""
                    critique_text = last_critique_text(crit_entry)
                    history = debate.get("history", [])
                    if not question and not critique_text and not history:
                        continue
                    alice_model = registry.resolve_model_name(debate.get("alice_model") or critic_slug)
                    bob_model = registry.resolve_model_name(debate.get("bob_model") or answer_slug)
                    tasks.append(
                        {
                            "id": f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}",
                            "type": "critique",
                            "mode": mode,
                            "question": question,
                            "answer": answer_text,
                            "critique": critique_text,
                            "debate": history,
                            "question_model": registry.display_name_for_slug(q_slug),
                            "answer_model": registry.display_name_for_slug(answer_slug),
                            "critic_model": registry.display_name_for_slug(critic_slug),
                            "alice_model": alice_model,
                            "bob_model": bob_model,
                            "run_id": debate.get("run_id") or crit_entry.get("run_id"),
                            "topic_slug": debate.get("topic_slug") or crit_entry.get("topic_slug"),
                        }
                    )
    return tasks


def build_illposed_prompt(task: Dict, guidance_q: str, registry) -> str:
    redactions, speaker_map = build_redactions(registry, task["alice_model"], task["bob_model"])
    question = redact_text(task.get("question", ""), redactions)
    answer = redact_text(task.get("answer", ""), redactions)
    debate = format_debate(task.get("debate", []), redactions, speaker_map)
    return (
        "You are a neutral judge. Decide whether the math question is ill-posed based on the debate.\n"
        "Alice claims the question is ill-posed. Bob defends that it is well-posed.\n"
        "Use the guidance below and ignore any party identity cues.\n"
        "Note: you are judging whether the claim of ill-posedness is valid. If the question is indeed ill-posed, but Bob gives a wrong reason, you should use \"ill-posed but wrong reason\".\n\n"
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
        '  "verdict": "ill-posed" | "not ill-posed" | "ill-posed but wrong reason" | "unknown" | "invalid",\n'
        '  "confidence": 1-5,\n'
        '  "comment": "<short justification>"\n'
        "}\n"
    )


def build_critique_prompt(task: Dict, guidance_a: str, guidance_c: str, registry) -> str:
    redactions, speaker_map = build_redactions(registry, task["alice_model"], task["bob_model"])
    question = redact_text(task.get("question", ""), redactions)
    answer = redact_text(task.get("answer", ""), redactions)
    critique = redact_text(task.get("critique", ""), redactions)
    debate = format_debate(task.get("debate", []), redactions, speaker_map)
    return (
        "You are a neutral judge. Decide whether Alice's critique is correct.\n"
        "Alice claims Bob's answer is wrong. Bob defends the answer.\n"
        "Use the guidance below and ignore any party identity cues.\n"
        "Note: you are judging whether the critique is valid. If the answer is indeed wrong, but Alice gives a wrong reason, you should use \"incorrect answer but wrong reason\".\n\n"
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
        '  "verdict": "correct answer" | "incorrect answer" | "incorrect answer but wrong reason" | "unknown" | "invalid",\n'
        '  "confidence": 1-5,\n'
        '  "comment": "<short justification>"\n'
        "}\n"
    )


def normalize_illposed_verdict(verdict: Optional[str]) -> str:
    if not verdict:
        return "unknown"
    v = verdict.strip().lower()
    if v in {"ill-posed", "not ill-posed", "ill-posed but wrong reason", "unknown", "invalid"}:
        return v
    if v in {"ill posed", "ill posed."}:
        return "ill-posed"
    if v in {"well-posed", "well posed", "not ill posed"}:
        return "not ill-posed"
    if "wrong reason" in v and "ill" in v:
        return "ill-posed but wrong reason"
    if v in {"uncertain", "unsure", "cannot determine"}:
        return "unknown"
    if v in {"n/a", "na"}:
        return "invalid"
    return "unknown"


def normalize_critique_verdict(verdict: Optional[str]) -> str:
    if not verdict:
        return "unknown"
    v = verdict.strip().lower()
    if v in {"correct answer", "incorrect answer", "incorrect answer but wrong reason", "unknown", "invalid"}:
        return v
    if v in {"right", "true"}:
        return "correct answer"
    if v in {"wrong", "false"}:
        return "incorrect answer"
    if "wrong reason" in v or "partially" in v:
        return "incorrect answer but wrong reason"
    if v in {"uncertain", "unsure", "cannot determine"}:
        return "unknown"
    if v in {"n/a", "na"}:
        return "invalid"
    return "unknown"


def parse_confidence(raw) -> int:
    try:
        conf = int(raw)
    except Exception:
        return 3
    return max(1, min(5, conf))


def parse_judgment(text: str, task: Dict) -> Dict:
    schema_hint = (
        '{"verdict": "...", "confidence": 1, "comment": "..."}'
    )
    parsed = safe_load_json(text or "", schema_hint=schema_hint)
    verdict = None
    confidence = None
    comment = ""
    if isinstance(parsed, dict):
        verdict = parsed.get("verdict")
        confidence = parse_confidence(parsed.get("confidence", confidence))
        comment = parsed.get("comment") or parsed.get("notes") or parsed.get("rationale") or ""
    if task["type"] == "illposed":
        verdict = normalize_illposed_verdict(verdict)
    else:
        verdict = normalize_critique_verdict(verdict)
    status = "succeeded" if verdict not in {"unknown", "invalid"} else "failed"
    return {
        "id": task["id"],
        "type": task["type"],
        "mode": task.get("mode"),
        "question_model": task.get("question_model"),
        "answer_model": task.get("answer_model"),
        "critic_model": task.get("critic_model"),
        "verdict": verdict,
        "confidence": confidence,
        "comment": comment,
        "status": status,
        "raw_response": text,
        "run_id": task.get("run_id"),
        "topic_slug": task.get("topic_slug"),
    }


def chunked(items: List[Dict], size: Optional[int]) -> Iterable[List[Dict]]:
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
    args = parser.parse_args()

    registry = load_registry(str(args.config))
    guidance_q = load_question_guidance()
    guidance_a = load_answer_guidance()
    guidance_c = load_critique_guidance()

    tasks: List[Dict] = []
    if args.mode in {"illposed", "all"}:
        tasks.extend(gather_illposed_tasks(args.debates_dir, args.answers_dir, args.benchmark_dir, registry))
    if args.mode in {"critiques", "all"}:
        tasks.extend(
            gather_critique_tasks(
                args.debates_dir,
                args.critiques_dir,
                args.answers_dir,
                args.benchmark_dir,
                registry,
            )
        )

    judges = registry.pick(args.models) if args.models else list(registry.models.values())
    jobs_by_judge: Dict[str, Dict] = {spec.name: {"spec": spec, "tasks": []} for spec in judges}

    for task in tasks:
        participants = {task.get("alice_model"), task.get("bob_model")}
        for spec in judges:
            if spec.name in participants:
                continue
            jobs_by_judge[spec.name]["tasks"].append(task)

    def process_judge(spec: ModelSpec, tasks_for_judge: List[Dict]) -> int:
        out_path = args.output_dir / f"{spec.slug}.json"
        decisions = load_decisions(out_path)
        pending = []
        for task in tasks_for_judge:
            if not args.overwrite and task["id"] in decisions:
                continue
            pending.append(task)
            if args.limit is not None and len(pending) >= args.limit:
                break
        if not pending:
            return 0

        for batch in chunked(pending, args.batch_size):
            prompts = []
            for task in batch:
                if task["type"] == "illposed":
                    prompt = build_illposed_prompt(task, guidance_q, registry)
                else:
                    prompt = build_critique_prompt(task, guidance_a, guidance_c, registry)
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
                print(f"Judge batch failed for {spec.name}: {exc}")
                continue
            for task, response in zip(batch, responses):
                decision = parse_judgment(response, task)
                decision["judge_model"] = spec.pretty
                decisions[task["id"]] = decision
            save_decisions(out_path, decisions)
            print(f"{spec.pretty}: saved {len(batch)} evaluations")
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
                print(f"Judge worker failed: {exc}")


if __name__ == "__main__":
    main()
