from pathlib import Path

from utils import load_json


def main():
    critiques_dir = Path("critiques")
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                entries = load_json(crit_file, [])
                for idx, entry in enumerate(entries):
                    attempts = entry.get("attempts") or []
                    verdict = attempts[-1].get("verdict") if attempts else None
                    if verdict in {None, "unknown"}:
                        print(
                            f"{mode_dir.name}/{q_dir.name}/{crit_file.stem} idx={idx} verdict={verdict} "
                            f"question_author={entry.get('question_author')} critic={entry.get('critic')}"
                        )


if __name__ == "__main__":
    main()
