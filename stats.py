from collections import Counter, defaultdict
from pathlib import Path

from utils import load_json


def count_human_labels(evaluations_dir: Path):
    label_counts = defaultdict(int)
    for eval_file in evaluations_dir.glob("*.json"):
        data = load_json(eval_file, {"decisions": []})
        for dec in data.get("decisions", []):
            label_counts[dec.get("id")] += 1
    return label_counts


def count_items(path: Path, kind: str):
    counts = 0
    if kind == "questions":
        for file in path.glob("*.json"):
            entries = load_json(file, [])
            counts += len(entries)
    elif kind == "answers":
        for q_dir in path.glob("*"):
            for ans_file in q_dir.glob("*.json"):
                entries = load_json(ans_file, [])
                counts += len(entries)
    elif kind in {"critiques", "illposed"}:
        base = "critiques" if kind == "critiques" else "debates/illposed"
        base_path = Path(base)
        for sub in base_path.glob("**/*.json"):
            entries = load_json(sub, [])
            counts += len(entries)
    return counts


def collect_claim_ids(critiques_dir: Path, debates_dir: Path):
    claim_ids = set()
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                q_slug = q_dir.name
                crit_ids = load_json(crit_file, [])
                for idx, _ in enumerate(crit_ids):
                    claim_ids.add(f"critique/{mode_dir.name}/{q_slug}/{crit_file.stem}/{idx}")
    for debate_file in (debates_dir / "illposed").glob("*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        debates = load_json(debate_file, [])
        for idx, _ in enumerate(debates):
            claim_ids.add(f"illposed/{q_slug}/{a_slug}/{idx}")
    return claim_ids


def critique_verdicts(critiques_dir: Path):
    verdict_counts = Counter()
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                entries = load_json(crit_file, [])
                for entry in entries:
                    attempts = entry.get("attempts") or []
                    if not attempts:
                        verdict_counts["missing"] += 1
                        continue
                    verdict = attempts[-1].get("verdict")
                    if not verdict:
                        verdict_counts["missing"] += 1
                    else:
                        verdict_counts[verdict] += 1
    return verdict_counts


def count_illposed_answers(answers_dir: Path):
    """Count answers with status='ill-posed'"""
    count = 0
    for q_dir in answers_dir.glob("*"):
        for ans_file in q_dir.glob("*.json"):
            entries = load_json(ans_file, [])
            for entry in entries:
                if entry.get("status") == "ill-posed":
                    count += 1
    return count


def main():
    benchmarks_dir = Path("benchmarks")
    answers_dir = Path("answers")
    critiques_dir = Path("critiques")
    debates_dir = Path("debates")
    evaluations_dir = Path("evaluations")

    questions = count_items(benchmarks_dir, "questions")
    answers = count_items(answers_dir, "answers")
    critiques_count = count_items(critiques_dir, "critiques")
    illposed_debate_count = count_items(Path(debates_dir), "illposed")
    illposed_answer_count = count_illposed_answers(answers_dir)

    labels = count_human_labels(evaluations_dir)
    claim_ids = collect_claim_ids(critiques_dir, debates_dir)

    # Only count labels for critique claims with non-correct verdicts and all illposed claims
    filtered_claim_ids = set()
    verdicts = critique_verdicts(critiques_dir)
    # Build a quick lookup for critique verdicts per id
    critique_verdict_map = {}
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                q_slug = q_dir.name
                entries = load_json(crit_file, [])
                for idx, entry in enumerate(entries):
                    attempts = entry.get("attempts") or []
                    verdict = attempts[-1].get("verdict") if attempts else None
                    cid = f"critique/{mode_dir.name}/{q_slug}/{crit_file.stem}/{idx}"
                    critique_verdict_map[cid] = verdict

    for cid in claim_ids:
        if cid.startswith("critique/"):
            v = critique_verdict_map.get(cid)
            if v == "correct":
                continue
        filtered_claim_ids.add(cid)

    label_hist = Counter(labels.get(cid, 0) for cid in filtered_claim_ids)

    print("Counts:")
    print(f"- Questions: {questions}")
    print(f"- Answers: {answers}")
    print(f"- Answers claiming ill-posed: {illposed_answer_count}")
    print(f"- Critiques: {critiques_count}")
    print(f"- Ill-posed debates: {illposed_debate_count}")
    v_counts = critique_verdicts(critiques_dir)
    print("\nCritiques by final verdict (including missing/unknown):")
    for verdict, count in v_counts.items():
        print(f"  {verdict}: {count}")
    print("\nLabel histogram (number of labels -> count of claims):")
    for n_labels in sorted(label_hist):
        print(f"  {n_labels}: {label_hist[n_labels]}")


if __name__ == "__main__":
    main()
