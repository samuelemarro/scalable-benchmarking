import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def list_illposed(answers_dir: Path, debates_dir: Path) -> List[Dict]:
    items = []
    for q_dir in answers_dir.glob("*"):
        q_slug = q_dir.name
        for answer_file in q_dir.glob("*.json"):
            a_slug = answer_file.stem
            records = load_json(answer_file, [])
            debate_file = debates_dir / "illposed" / q_slug / f"{a_slug}.json"
            debates = load_json(debate_file, [])
            for idx, rec in enumerate(records):
                if rec.get("status") != "ill-posed":
                    continue
                debate_history = debates[idx] if idx < len(debates) else {}
                items.append(
                    {
                        "key": f"{q_slug}/{a_slug}/{idx}",
                        "question": rec.get("question"),
                        "claim": rec.get("ill_posed_claim"),
                        "debate": debate_history,
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
                critiques = load_json(crit_file, [])
                debate_file = debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
                debates = load_json(debate_file, [])
                for idx, crit in enumerate(critiques):
                    if not crit or crit.get("status") != "succeeded":
                        continue
                    attempts = crit.get("attempts") or []
                    last_attempt = attempts[-1] if attempts else {}
                    debate_history = debates[idx] if idx < len(debates) else {}
                    items.append(
                        {
                            "key": f"{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}",
                            "question": crit.get("question"),
                            "critique": last_attempt.get("raw_critique"),
                            "verdict": last_attempt.get("verdict"),
                            "notes": last_attempt.get("notes"),
                            "critic": critic_slug,
                            "answer_author": answer_slug,
                            "debate": debate_history,
                        }
                    )
    return items


def record_evaluation(username: str, key: str, verdict: str, comment: str, section: str, store_dir: Path):
    eval_path = store_dir / f"{username}.json"
    payload = load_json(eval_path, {"ill_posed": {}, "critiques": {}})
    payload.setdefault(section, {})
    payload[section][key] = {"verdict": verdict, "comment": comment}
    save_json(eval_path, payload)


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
        if key.count("/") == 2:
            # ill-posed
            parts = key.split("/")
            q_slug, a_slug, idx = parts[0], parts[1], int(parts[2])
            answer_file = args.answers_dir / q_slug / f"{a_slug}.json"
            debate_file = args.debates_dir / "illposed" / q_slug / f"{a_slug}.json"
            answers = load_json(answer_file, [])
            debates = load_json(debate_file, [])
            item = {
                "key": key,
                "answer_record": answers[idx] if idx < len(answers) else {},
                "debate": debates[idx] if idx < len(debates) else {},
            }
            show_item(item)
        else:
            parts = key.split("/")
            if len(parts) != 4:
                raise ValueError("Critique key must look like mode/qslug/critic__answer/idx")
            mode, q_slug, pair, idx_str = parts
            critic_slug, answer_slug = pair.split("__")
            idx = int(idx_str)
            crit_file = args.critiques_dir / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
            debate_file = args.debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json"
            critiques = load_json(crit_file, [])
            debates = load_json(debate_file, [])
            item = {
                "key": key,
                "critique_record": critiques[idx] if idx < len(critiques) else {},
                "debate": debates[idx] if idx < len(debates) else {},
            }
            show_item(item)
        return

    if args.verdict:
        key = args.verdict.split(":")[0] if ":" in args.verdict else None
        if not key:
            raise ValueError("Provide verdict as key:decision (e.g., ill-posed/openai-gpt-5-2025-08-07__...:correct)")
        key_part, decision = args.verdict.split(":")
        section = "ill_posed" if key_part.count("/") == 2 else "critiques"
        record_evaluation(args.username, key_part, decision, args.comment, section, args.evaluations_dir)
        print(f"Recorded {decision} for {key_part}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
