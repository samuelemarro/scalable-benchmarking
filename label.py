import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from constants import STATUS_ILL_POSED, STATUS_SUCCEEDED
from data_models import (
    HumanEvaluation,
    HumanEvaluationFile,
    load_answer_entries,
    load_critique_entries,
    load_debate_entries,
    load_human_evaluation_entries,
    save_human_evaluation_entries,
)
from utils import entry_key


def _task_id(prefix: str, run_id: Optional[str], topic_slug: Optional[str], question: Optional[str]) -> str:
    if run_id:
        return f"{prefix}/{run_id}"
    if topic_slug and question:
        digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}/{topic_slug}/{digest}"
    return f"{prefix}/unknown"


def _digest_question(question: Optional[str]) -> Optional[str]:
    if not question:
        return None
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]


def _find_entry_by_run_id(entries: List, run_id: Optional[str]):
    if run_id is None:
        return None
    run_id = str(run_id)
    for entry in entries:
        if not entry:
            continue
        if str(entry.run_id) == run_id:
            return entry
    return None


def _find_entry_by_digest(entries: List, topic_slug: Optional[str], digest: Optional[str]):
    if not topic_slug or not digest:
        return None
    for entry in entries:
        if not entry or entry.topic_slug != topic_slug:
            continue
        if _digest_question(entry.question) == digest:
            return entry
    return None


def _maybe_index_fallback(entries: List, idx_str: Optional[str]):
    if idx_str is None:
        return None
    try:
        idx = int(idx_str)
    except ValueError:
        return None
    if 0 <= idx < len(entries):
        return entries[idx]
    return None


def list_illposed(answers_dir: Path, debates_dir: Path) -> List[Dict]:
    items = []
    for q_dir in answers_dir.glob("*"):
        q_slug = q_dir.name
        for answer_file in q_dir.glob("*.json"):
            a_slug = answer_file.stem
            records = load_answer_entries(answer_file)
            debate_file = debates_dir / "illposed" / q_slug / f"{a_slug}.json"
            debates = load_debate_entries(debate_file)
            debate_map: Dict = {}
            for debate in debates:
                if not debate:
                    continue
                key = entry_key(debate.run_id, debate.topic_slug, debate.question)
                if key and key not in debate_map:
                    debate_map[key] = debate
            for idx, rec in enumerate(records):
                if not rec or rec.status != STATUS_ILL_POSED:
                    continue
                key = entry_key(rec.run_id, rec.topic_slug, rec.question)
                debate_history = debate_map.get(key)
                prefix = f"illposed/{q_slug}/{a_slug}"
                item_key = _task_id(
                    prefix,
                    rec.run_id or (debate_history.run_id if debate_history else None),
                    rec.topic_slug or (debate_history.topic_slug if debate_history else None),
                    rec.question or (debate_history.question if debate_history else None),
                )
                items.append(
                    {
                        "key": item_key,
                        "question": rec.question,
                        "claim": rec.ill_posed_claim,
                        "debate": debate_history.model_dump(exclude_none=True) if debate_history else {},
                    }
                )
    return items


def list_critiques(critiques_dir: Path, debates_dir: Path) -> List[Dict]:
    items = []
    for mode_dir in critiques_dir.glob("*"):
        mode = mode_dir.name
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                critiques = load_critique_entries(crit_file)
                debate_file = debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                debates = load_debate_entries(debate_file)
                debate_map: Dict = {}
                for debate in debates:
                    if not debate:
                        continue
                    key = entry_key(debate.run_id, debate.topic_slug, debate.question)
                    if key and key not in debate_map:
                        debate_map[key] = debate
                for idx, crit in enumerate(critiques):
                    if not crit or crit.status != STATUS_SUCCEEDED:
                        continue
                    attempts = crit.attempts or []
                    last_attempt = attempts[-1] if attempts else None
                    key = entry_key(crit.run_id, crit.topic_slug, crit.question)
                    debate_history = debate_map.get(key)
                    prefix = f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}"
                    item_key = _task_id(
                        prefix,
                        crit.run_id or (debate_history.run_id if debate_history else None),
                        crit.topic_slug or (debate_history.topic_slug if debate_history else None),
                        crit.question or (debate_history.question if debate_history else None),
                    )
                    items.append(
                        {
                            "key": item_key,
                            "question": crit.question,
                            "critique": last_attempt.raw_critique if last_attempt else None,
                            "verdict": last_attempt.verdict if last_attempt else None,
                            "notes": last_attempt.notes if last_attempt else None,
                            "critic": critic_slug,
                            "answer_author": answer_slug,
                            "debate": debate_history.model_dump(exclude_none=True) if debate_history else {},
                        }
                    )
    return items


def record_evaluation(
    username: str,
    key: str,
    verdict: str,
    comment: str,
    eval_type: str,
    store_dir: Path,
    mode: Optional[str] = None,
):
    eval_path = store_dir / f"{username}.json"
    payload = load_human_evaluation_entries(eval_path)
    decisions = {decision.id: decision for decision in payload.decisions if decision.id}
    decisions[key] = HumanEvaluation(
        id=key,
        type=eval_type,
        mode=mode,
        verdict=verdict,
        comment=comment,
    )
    save_human_evaluation_entries(eval_path, HumanEvaluationFile(decisions=list(decisions.values())))


def show_item(item: Dict):
    print(json.dumps(item, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Human evaluation CLI for ill-posed claims and critiques.")
    parser.add_argument("--username", required=True, help="Evaluator name.")
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--evaluations-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--list", choices=["ill-posed", "critiques"], help="List pending items.")
    parser.add_argument("--show", help="Show a specific key.")
    parser.add_argument("--verdict", help="Record verdict for a key.")
    parser.add_argument("--comment", default="", help="Optional comment for verdict.")
    args = parser.parse_args()

    if args.list:
        if args.list == "ill-posed":
            items = list_illposed(args.answers_dir, args.debates_dir)
        else:
            items = list_critiques(args.critiques_dir, args.debates_dir)
        for item in items:
            print(item["key"])
        return

    if args.show:
        key = args.show
        parts = key.split("/")
        critique_modes = {p.name for p in args.critiques_dir.glob("*") if p.is_dir()}
        if parts[0] == "critique":
            parts = parts[1:]
        elif parts[0] == "illposed":
            parts = parts[1:]

        is_critique = parts[0] in critique_modes if parts else False
        if not is_critique:
            # ill-posed
            if len(parts) < 3:
                raise ValueError("Ill-posed key must look like qslug/aslug/run_id or qslug/aslug/topic/hash")
            q_slug, a_slug = parts[0], parts[1]
            run_id = parts[2] if len(parts) == 3 else None
            topic_slug = parts[2] if len(parts) >= 4 else None
            digest = parts[3] if len(parts) >= 4 else None
            answer_file = args.answers_dir / q_slug / f"{a_slug}.json"
            debate_file = args.debates_dir / "illposed" / q_slug / f"{a_slug}.json"
            answers = load_answer_entries(answer_file)
            debates = load_debate_entries(debate_file)
            answer_record = _find_entry_by_run_id(answers, run_id) or _find_entry_by_digest(
                answers, topic_slug, digest
            )
            if not answer_record and run_id:
                answer_record = _maybe_index_fallback(answers, run_id)
            debate_record = _find_entry_by_run_id(debates, run_id) or _find_entry_by_digest(
                debates, topic_slug, digest
            )
            if not debate_record and run_id:
                debate_record = _maybe_index_fallback(debates, run_id)
            item = {
                "key": key,
                "answer_record": answer_record.model_dump(exclude_none=True) if answer_record else {},
                "debate": debate_record.model_dump(exclude_none=True) if debate_record else {},
            }
            show_item(item)
        else:
            # critique
            if len(parts) < 4:
                raise ValueError("Critique key must look like mode/qslug/critic__answer/run_id or mode/qslug/critic__answer/topic/hash")
            mode, q_slug, pair = parts[0], parts[1], parts[2]
            critic_slug, answer_slug = pair.split("__")
            run_id = parts[3] if len(parts) == 4 else None
            topic_slug = parts[3] if len(parts) >= 5 else None
            digest = parts[4] if len(parts) >= 5 else None
            crit_file = args.critiques_dir / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
            debate_file = args.debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
            critiques = load_critique_entries(crit_file)
            debates = load_debate_entries(debate_file)
            critique_record = _find_entry_by_run_id(critiques, run_id) or _find_entry_by_digest(
                critiques, topic_slug, digest
            )
            if not critique_record and run_id:
                critique_record = _maybe_index_fallback(critiques, run_id)
            debate_record = _find_entry_by_run_id(debates, run_id) or _find_entry_by_digest(
                debates, topic_slug, digest
            )
            if not debate_record and run_id:
                debate_record = _maybe_index_fallback(debates, run_id)
            item = {
                "key": key,
                "critique_record": critique_record.model_dump(exclude_none=True) if critique_record else {},
                "debate": debate_record.model_dump(exclude_none=True) if debate_record else {},
            }
            show_item(item)
        return

    if args.verdict:
        key = args.verdict.split(":")[0] if ":" in args.verdict else None
        if not key:
            raise ValueError("Provide verdict as key:decision (e.g., ill-posed/openai-gpt-5-2025-08-07__...:correct)")
        key_part, decision = args.verdict.split(":")
        eval_type = "illposed" if key_part.count("/") == 2 else "critique"
        mode = None
        if eval_type == "critique":
            parts = key_part.split("/")
            mode = parts[0] if len(parts) == 4 else None
        record_evaluation(args.username, key_part, decision, args.comment, eval_type, args.evaluations_dir, mode=mode)
        print(f"Recorded {decision} for {key_part}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
