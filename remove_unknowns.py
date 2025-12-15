import json
from pathlib import Path

from utils import load_json, save_json


def last_verdict_attempt(attempts):
    if not attempts:
        return None
    return attempts[-1].get("verdict")


def clean_benchmarks(path: Path):
    for file in path.glob("*.json"):
        entries = load_json(file, [])
        new_entries = []
        for entry in entries:
            gens = entry.get("generation_rounds") or []
            if gens:
                refinements = gens[-1].get("refinement_rounds") or []
                verdict = refinements[-1].get("evaluation", {}).get("verdict") if refinements else None
                if verdict == "unknown":
                    continue
            new_entries.append(entry)
        if len(new_entries) != len(entries):
            save_json(file, new_entries)


def clean_answers(path: Path):
    for q_dir in path.glob("*"):
        for ans_file in q_dir.glob("*.json"):
            entries = load_json(ans_file, [])
            changed = False
            new_entries = []
            for entry in entries:
                verdict = last_verdict_attempt(entry.get("attempts") or [])
                if verdict == "unknown":
                    changed = True
                    continue
                new_entries.append(entry)
            if changed:
                save_json(ans_file, new_entries)


def clean_critiques(path: Path):
    for mode_dir in path.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                entries = load_json(crit_file, [])
                changed = False
                new_entries = []
                for entry in entries:
                    verdict = last_verdict_attempt(entry.get("attempts") or [])
                    if verdict == "unknown":
                        changed = True
                        continue
                    new_entries.append(entry)
                if changed:
                    save_json(crit_file, new_entries)


def main():
    clean_benchmarks(Path("benchmarks"))
    clean_answers(Path("answers"))
    clean_critiques(Path("critiques"))
    print("Cleaned entries with final verdict 'unknown'.")


if __name__ == "__main__":
    main()
