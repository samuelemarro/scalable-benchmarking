import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from model_config import _slugify, load_registry
from constants import CRITIQUE_VERDICT_CORRECT


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


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


def gather_illposed(debates_dir: Path, answers_dir: Path, benchmark_dir: Path, registry) -> List[Dict]:
    items = []
    for debate_file in debates_dir.glob("illposed/*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        question_model = registry.display_name_for_slug(q_slug)
        answer_model = registry.display_name_for_slug(a_slug)
        debates = load_json(debate_file, [])
        answers = load_json(answers_dir / q_slug / f"{a_slug}.json", [])
        max_len = max(len(debates), len(answers))
        for idx in range(max_len):
            debate = debates[idx] if idx < len(debates) else {}
            question = debate.get("question") or ""
            answer_record = answers[idx] if idx < len(answers) else {}
            answer_text = final_answer(answer_record) or answer_record.get("answer") or ""
            if not answer_text:
                answer_text = benchmark_answer_for_index(benchmark_dir, q_slug, idx) or ""
            items.append(
                {
                    "id": f"illposed/{q_slug}/{a_slug}/{idx}",
                    "type": "illposed",
                    "question_model": question_model,
                    "answer_model": answer_model,
                    "critic_model": answer_model,
                    "topic": question[:80] + ("..." if len(question) > 80 else ""),
                    "run_id": debate.get("run_id"),
                    "topic_slug": debate.get("topic_slug"),
                    "question": question,
                    "answer": answer_text,
                    "debate": debate.get("history", []),
                }
            )
    return items


def gather_critiques(debates_dir: Path, critiques_dir: Path, answers_dir: Path, benchmark_dir: Path, registry) -> List[Dict]:
    items = []
    for crit_mode_dir in critiques_dir.glob("*"):
        mode = crit_mode_dir.name
        for q_dir in crit_mode_dir.glob("*"):
            q_slug = q_dir.name
            question_model = registry.display_name_for_slug(q_slug)
            for crit_file in q_dir.glob("*.json"):
                critic_slug, answer_slug = crit_file.stem.split("__")
                critic_model = registry.display_name_for_slug(critic_slug)
                answer_model = registry.display_name_for_slug(answer_slug)

                critiques = load_json(crit_file, [])
                answers = load_json(answers_dir / q_slug / f"{answer_slug}.json", [])
                debates = load_json(debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json", [])

                max_len = len(critiques)
                for idx in range(max_len):
                    critique_entry = critiques[idx]
                    debate = debates[idx] if idx < len(debates) else {}
                    question = debate.get("question") or critique_entry.get("question", "")
                    answer_record = answers[idx] if idx < len(answers) else {}
                    answer_text = final_answer(answer_record) or answer_record.get("answer") or ""
                    if not answer_text:
                        answer_text = benchmark_answer_for_index(benchmark_dir, q_slug, idx) or ""
                    critique_attempts = critique_entry.get("attempts") or []
                    if critique_attempts and critique_attempts[-1].get("verdict") == CRITIQUE_VERDICT_CORRECT:
                        continue
                    critique_text = critique_attempts[-1].get("raw_critique") if critique_attempts else ""
                    items.append(
                        {
                            "id": f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}",
                            "type": "critique",
                            "mode": mode,
                            "question_model": question_model,
                            "answer_model": answer_model,
                            "critic_model": critic_model,
                            "topic": question[:80] + ("..." if len(question) > 80 else ""),
                            "run_id": debate.get("run_id") or critique_entry.get("run_id"),
                            "topic_slug": debate.get("topic_slug") or critique_entry.get("topic_slug"),
                            "question": question,
                            "answer": answer_text,
                            "critique": critique_text,
                            "debate": debate.get("history", []),
                        }
                    )
    return items


def load_evaluations(path: Path) -> Dict[str, Dict]:
    data = load_json(path, {"decisions": []})
    by_id = {d.get("id"): d for d in data.get("decisions", [])}
    return by_id


def save_evaluation(path: Path, decisions: Dict[str, Dict]):
    payload = {"decisions": list(decisions.values())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def render_task(task: Dict, existing: Optional[Dict], save_cb):
    st.markdown(f"### {task['id']}")
    st.write(f"**Question model:** {task['question_model']}  |  **Answer model:** {task['answer_model']}  |  **Critic:** {task.get('critic_model','')}")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Question")
        st.write(task.get("question", ""))
        st.subheader("Answer")
        st.write(task.get("answer", ""))
    with cols[1]:
        if task["type"] == "critique":
            st.subheader("Critique")
            st.write(task.get("critique", ""))
        st.subheader("Debate")
        for msg in task.get("debate", []):
            st.markdown(f"- **{msg.get('speaker','')}**: {msg.get('message','')}")

    st.divider()
    if task["type"] == "illposed":
        options = [
            "ill-posed",
            "not ill-posed",
            "ill-posed but wrong reason",
            "unknown",
            "invalid",
        ]
    else:
        options = [
            "incorrect",
            "correct",
            "incorrect but wrong reason",
            "unknown",
            "invalid",
        ]

    existing_choice = existing.get("verdict") if existing else None
    choice = st.radio("Verdict", options, index=options.index(existing_choice) if existing_choice in options else 0, key=f"verdict-{task['id']}")
    confidence = st.slider("Confidence (1-5)", 1, 5, int(existing.get("confidence", 3) if existing else 3), 1, key=f"conf-{task['id']}")
    comment = st.text_area("Comments", value=existing.get("comment", "") if existing else "", key=f"comment-{task['id']}")
    if st.button("Save", key=f"save-{task['id']}"):
        save_cb(
            {
                "id": task["id"],
                "type": task["type"],
                "question_model": task["question_model"],
                "answer_model": task["answer_model"],
                "critic_model": task.get("critic_model"),
                "verdict": choice,
                "confidence": confidence,
                "comment": comment,
            }
        )
        st.success("Saved")


def main():
    parser = argparse.ArgumentParser(description="Interactive web labeller")
    parser.add_argument("--username", required=True)
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--evaluations-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    args, _ = parser.parse_known_args()

    registry = load_registry(str(args.config))

    illposed = gather_illposed(args.debates_dir, args.answers_dir, args.benchmark_dir, registry)
    critiques = gather_critiques(args.debates_dir, args.critiques_dir, args.answers_dir, args.benchmark_dir, registry)
    tasks = illposed + critiques

    eval_path = args.evaluations_dir / f"{args.username}.json"
    decisions = load_evaluations(eval_path)

    st.title(f"Labeller - {args.username}")

    claim_filter = st.multiselect(
        "Claim type",
        ["illposed", "contradictor", "evaluator"],
        default=["illposed", "contradictor", "evaluator"],
    )
    questioners = sorted({t["question_model"] for t in tasks})
    answerers = sorted({t["answer_model"] for t in tasks})
    critics = sorted({t.get("critic_model") for t in tasks if t.get("critic_model")})
    selected_q = st.multiselect("Question model", questioners, default=questioners)
    selected_a = st.multiselect("Answer model", answerers, default=answerers)
    selected_c = st.multiselect("Critic", critics, default=critics)
    only_unlabeled = st.checkbox("Only not yet labelled by me", value=True)

    filtered = []
    for task in tasks:
        task_kind = task["type"] if task["type"] == "illposed" else task.get("mode", "critique")
        if task_kind not in claim_filter:
            continue
        if task["question_model"] not in selected_q:
            continue
        if task["answer_model"] not in selected_a:
            continue
        if task.get("critic_model") and selected_c and task["critic_model"] not in selected_c:
            continue
        already = decisions.get(task["id"])
        if only_unlabeled and already:
            continue
        filtered.append((task, already))

    st.subheader(f"Tasks ({len(filtered)})")

    def save_decision(decision: Dict):
        decisions[decision["id"]] = decision
        save_evaluation(eval_path, decisions)

    selected_task_id = st.session_state.get("selected_task_id")

    if not selected_task_id:
        for task, existing in filtered:
            claimant = task.get("critic_model") or task.get("answer_model")
            adversary = task.get("answer_model") if task["type"] == "critique" else task.get("question_model")
            topic = task.get("topic", "")
            cols = st.columns([4, 2])
            with cols[0]:
                st.markdown(f"**{task['type']}** | claimant: {claimant} | adversary: {adversary} | topic: {topic}")
            with cols[1]:
                if st.button("Open", key=f"open-{task['id']}"):
                    st.session_state["selected_task_id"] = task["id"]
                    st.rerun()
        if not filtered:
            st.info("No tasks match the current filters.")
    else:
        task, existing = next(((t, e) for t, e in filtered if t["id"] == selected_task_id), (None, None))
        if not task:
            st.warning("Selected task not found in current filter. Clear selection.")
        else:
            if st.button("Back to list"):
                st.session_state["selected_task_id"] = None
                st.rerun()
            st.divider()
            render_task(task, existing, save_decision)


if __name__ == "__main__":
    main()
