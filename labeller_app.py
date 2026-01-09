import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from model_config import load_registry
from constants import CRITIQUE_VERDICT_CORRECT
from data_models import (
    AnswerEntry,
    HumanEvaluation,
    HumanEvaluationFile,
    JudgingTask,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_human_evaluation_entries,
    save_human_evaluation_entries,
)
from utils import benchmark_answers_from_entries, entry_key


def _task_id(prefix: str, run_id: Optional[str], topic_slug: Optional[str], question: Optional[str]) -> str:
    if run_id:
        return f"{prefix}/{run_id}"
    if topic_slug and question:
        digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}/{topic_slug}/{digest}"
    return f"{prefix}/unknown"


def final_answer(entry: AnswerEntry) -> Optional[str]:
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].answer


def benchmark_answer_map(benchmark_dir: Path, q_slug: str) -> Dict:
    bench_path = benchmark_dir / f"{q_slug}.json"
    entries = load_benchmark_entries(bench_path)
    answers = benchmark_answers_from_entries(q_slug, entries)
    mapping: Dict = {}
    for entry in answers:
        if not entry:
            continue
        key = entry_key(entry.run_id, entry.topic_slug, entry.question)
        if key and key not in mapping:
            mapping[key] = final_answer(entry) or ""
    return mapping


def gather_illposed(debates_dir: Path, answers_dir: Path, benchmark_dir: Path, registry) -> List[JudgingTask]:
    items = []
    for debate_file in debates_dir.glob("illposed/*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        question_model = registry.display_name_for_slug(q_slug)
        answer_model = registry.display_name_for_slug(a_slug)
        debates = load_debate_entries(debate_file)
        answers = load_answer_entries(answers_dir / q_slug / f"{a_slug}.json")
        benchmark_answers = benchmark_answer_map(benchmark_dir, q_slug)
        debate_map: Dict = {}
        for debate in debates:
            if not debate:
                continue
            key = entry_key(debate.run_id, debate.topic_slug, debate.question)
            if key and key not in debate_map:
                debate_map[key] = debate
        answer_map: Dict = {}
        for answer in answers:
            if not answer:
                continue
            key = entry_key(answer.run_id, answer.topic_slug, answer.question)
            if key and key not in answer_map:
                answer_map[key] = answer
        keys = set(debate_map) | set(answer_map) | set(benchmark_answers)
        for key in keys:
            debate = debate_map.get(key)
            question = debate.question if debate else ""
            answer_record = answer_map.get(key)
            answer_text = final_answer(answer_record) if answer_record else ""
            if not answer_text:
                answer_text = benchmark_answers.get(key, "")
            run_id = (debate.run_id if debate else None) or (answer_record.run_id if answer_record else None)
            topic_slug = (debate.topic_slug if debate else None) or (answer_record.topic_slug if answer_record else None)
            prefix = f"illposed/{q_slug}/{a_slug}"
            item_id = _task_id(prefix, run_id, topic_slug, question or (answer_record.question if answer_record else None))
            items.append(
                JudgingTask(
                    id=item_id,
                    type="illposed",
                    question_model=question_model,
                    answer_model=answer_model,
                    critic_model=answer_model,
                    run_id=run_id,
                    topic_slug=topic_slug,
                    question=question or (answer_record.question if answer_record else ""),
                    answer=answer_text,
                    debate_history=debate.history if debate else [],
                )
            )
    return items


def gather_critiques(debates_dir: Path, critiques_dir: Path, answers_dir: Path, benchmark_dir: Path, registry) -> List[JudgingTask]:
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

                critiques = load_critique_entries(crit_file)
                answers = load_answer_entries(answers_dir / q_slug / f"{answer_slug}.json")
                debates = load_debate_entries(debates_dir / "critiques" / mode / q_slug / f"{critic_slug}__{answer_slug}.json")
                benchmark_answers = benchmark_answer_map(benchmark_dir, q_slug)
                debate_map: Dict = {}
                for debate in debates:
                    if not debate:
                        continue
                    key = entry_key(debate.run_id, debate.topic_slug, debate.question)
                    if key and key not in debate_map:
                        debate_map[key] = debate
                answer_map: Dict = {}
                for answer in answers:
                    if not answer:
                        continue
                    key = entry_key(answer.run_id, answer.topic_slug, answer.question)
                    if key and key not in answer_map:
                        answer_map[key] = answer

                for critique_entry in critiques:
                    if not critique_entry:
                        continue
                    critique_attempts = critique_entry.attempts or []
                    if critique_attempts and critique_attempts[-1].verdict == CRITIQUE_VERDICT_CORRECT:
                        continue
                    key = entry_key(critique_entry.run_id, critique_entry.topic_slug, critique_entry.question)
                    debate = debate_map.get(key)
                    question = (debate.question if debate else "") or critique_entry.question
                    answer_record = answer_map.get(key)
                    answer_text = final_answer(answer_record) if answer_record else ""
                    if not answer_text:
                        answer_text = benchmark_answers.get(key, "")
                    critique_text = critique_attempts[-1].raw_critique if critique_attempts else ""
                    run_id = (debate.run_id if debate else None) or critique_entry.run_id
                    topic_slug = (debate.topic_slug if debate else None) or critique_entry.topic_slug
                    prefix = f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}"
                    item_id = _task_id(prefix, run_id, topic_slug, question)
                    items.append(
                        JudgingTask(
                            id=item_id,
                            type="critique",
                            mode=mode,
                            question_model=question_model,
                            answer_model=answer_model,
                            critic_model=critic_model,
                            run_id=run_id,
                            topic_slug=topic_slug,
                            question=question,
                            answer=answer_text,
                            critique=critique_text,
                            debate_history=debate.history if debate else [],
                        )
                    )
    return items


def load_evaluations(path: Path) -> Dict[str, HumanEvaluation]:
    payload = load_human_evaluation_entries(path)
    return {d.id: d for d in payload.decisions if d.id}


def save_evaluation(path: Path, decisions: Dict[str, HumanEvaluation]):
    save_human_evaluation_entries(path, HumanEvaluationFile(decisions=list(decisions.values())))


def render_task(task: JudgingTask, existing: Optional[HumanEvaluation], save_cb):
    st.markdown(f"### {task.id}")
    st.write(f"**Question model:** {task.question_model}  |  **Answer model:** {task.answer_model}  |  **Critic:** {task.critic_model or ''}")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Question")
        st.write(task.question or "")
        st.subheader("Answer")
        st.write(task.answer or "")
    with cols[1]:
        if task.type == "critique":
            st.subheader("Critique")
            st.write(task.critique or "")
        st.subheader("Debate")
        for msg in task.debate_history or []:
            st.markdown(f"- **{msg.speaker}**: {msg.message}")

    st.divider()
    if task.type == "illposed":
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

    existing_choice = existing.verdict if existing else None
    choice = st.radio("Verdict", options, index=options.index(existing_choice) if existing_choice in options else 0, key=f"verdict-{task.id}")
    confidence = st.slider("Confidence (1-5)", 1, 5, int(existing.confidence if existing and existing.confidence is not None else 3), 1, key=f"conf-{task.id}")
    comment = st.text_area("Comments", value=existing.comment if existing else "", key=f"comment-{task.id}")
    if st.button("Save", key=f"save-{task.id}"):
        save_cb(
            HumanEvaluation(
                id=task.id,
                type=task.type,
                mode=task.mode,
                question_model=task.question_model,
                answer_model=task.answer_model,
                critic_model=task.critic_model,
                verdict=choice,
                confidence=confidence,
                comment=comment,
            )
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
    questioners = sorted({t.question_model for t in tasks if t.question_model})
    answerers = sorted({t.answer_model for t in tasks if t.answer_model})
    critics = sorted({t.critic_model for t in tasks if t.critic_model})
    selected_q = st.multiselect("Question model", questioners, default=questioners)
    selected_a = st.multiselect("Answer model", answerers, default=answerers)
    selected_c = st.multiselect("Critic", critics, default=critics)
    only_unlabeled = st.checkbox("Only not yet labelled by me", value=True)

    filtered = []
    for task in tasks:
        task_kind = task.type if task.type == "illposed" else task.mode or "critique"
        if task_kind not in claim_filter:
            continue
        if task.question_model not in selected_q:
            continue
        if task.answer_model not in selected_a:
            continue
        if task.critic_model and selected_c and task.critic_model not in selected_c:
            continue
        already = decisions.get(task.id)
        if only_unlabeled and already:
            continue
        filtered.append((task, already))

    st.subheader(f"Tasks ({len(filtered)})")

    def save_decision(decision: HumanEvaluation):
        decisions[decision.id] = decision
        save_evaluation(eval_path, decisions)

    selected_task_id = st.session_state.get("selected_task_id")

    if not selected_task_id:
        for task, existing in filtered:
            claimant = task.critic_model or task.answer_model
            adversary = task.answer_model if task.type == "critique" else task.question_model
            topic = (task.question or "")[:80] + ("..." if task.question and len(task.question) > 80 else "")
            cols = st.columns([4, 2])
            with cols[0]:
                st.markdown(f"**{task.type}** | claimant: {claimant} | adversary: {adversary} | topic: {topic}")
            with cols[1]:
                if st.button("Open", key=f"open-{task.id}"):
                    st.session_state["selected_task_id"] = task.id
                    st.rerun()
        if not filtered:
            st.info("No tasks match the current filters.")
    else:
        task, existing = next(((t, e) for t, e in filtered if t.id == selected_task_id), (None, None))
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
