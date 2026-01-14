import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from model_config import load_registry
from constants import (
    VALID_CRITIQUE_DEBATE_VERDICTS,
    VALID_ILLPOSED_DEBATE_VERDICTS,
)
from prompt_library import (
    load_answer_guidance,
    load_critique_guidance,
    load_judgment_critique_guidance,
    load_judgment_illposed_guidance,
    load_question_guidance,
)
from data_models import (
    AnswerEntry,
    AutomatedEvaluation,
    HumanEvaluation,
    HumanEvaluationFile,
    JudgingTask,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_evaluation_entries,
    load_human_evaluation_entries,
    save_human_evaluation_entries,
)
from utils import benchmark_answers_from_entries

st.set_page_config(layout="wide")


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


def load_evaluations(path: Path) -> Dict[str, HumanEvaluation]:
    payload = load_human_evaluation_entries(path)
    return {d.id: d for d in payload.decisions if d.id}


def load_other_evaluations(evaluations_dir: Path, username: str) -> Dict[str, List[HumanEvaluation]]:
    collected: Dict[str, List[HumanEvaluation]] = {}
    if not evaluations_dir.exists():
        return collected
    for eval_file in evaluations_dir.glob("*.json"):
        if eval_file.stem == username:
            continue
        payload = load_human_evaluation_entries(eval_file)
        for decision in payload.decisions:
            if not decision.id:
                continue
            collected.setdefault(decision.id, []).append(decision)
    return collected


def save_evaluation(path: Path, decisions: Dict[str, HumanEvaluation]):
    save_human_evaluation_entries(path, HumanEvaluationFile(decisions=list(decisions.values())))


def build_illposed_claim(entry: Optional[AnswerEntry]) -> str:
    if not entry:
        return ""
    attempts = entry.attempts or []
    if not attempts:
        return ""
    evaluation = attempts[-1].evaluation or {}
    issues = evaluation.get("issues")
    if not issues:
        return ""
    if isinstance(issues, str):
        issue_list = [issues]
    else:
        try:
            issue_list = [str(issue) for issue in issues if issue]
        except TypeError:
            issue_list = [str(issues)]
    return "\n".join(f"- {issue}" for issue in issue_list)


def run_git_sync(label_id: str, username: str, eval_path: Path) -> Tuple[bool, str, str]:
    repo_root = Path(__file__).resolve().parent
    try:
        rel_path = eval_path.relative_to(repo_root)
    except ValueError:
        rel_path = eval_path
    outputs: List[str] = []

    def run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
        if output:
            outputs.append(f"$ {' '.join(cmd)}\n{output}")
        return result

    pull = run(["git", "pull", "--no-rebase", "--autostash"])
    if pull.returncode != 0:
        return False, "Git pull failed", "\n\n".join(outputs)

    add = run(["git", "add", str(rel_path)])
    if add.returncode != 0:
        return False, "Git add failed", "\n\n".join(outputs)

    diff = run(["git", "diff", "--cached", "--quiet", "--", str(rel_path)])
    if diff.returncode == 0:
        return True, "Saved (no changes to commit)", "\n\n".join(outputs)
    if diff.returncode != 1:
        return False, "Git diff failed", "\n\n".join(outputs)

    commit = run(["git", "commit", "-m", f"Labelling {label_id} {username}", "--", str(rel_path)])
    if commit.returncode != 0:
        return False, "Git commit failed", "\n\n".join(outputs)

    push = run(["git", "push"])
    if push.returncode != 0:
        return False, "Git push failed", "\n\n".join(outputs)

    return True, "Saved and pushed", "\n\n".join(outputs)


def verdict_choices(task_type: str) -> List[str]:
    if task_type == "illposed":
        preferred = [
            "claimant_wins",
            "defender_wins_incorrect",
            "wrong_problem",
            "mixed",
            "unknown",
            "other",
        ]
        return [v for v in preferred if v in VALID_ILLPOSED_DEBATE_VERDICTS or v == "other"]
    preferred = [
        "claimant_wins",
        "defender_wins_incorrect",
        "defender_wins_minor",
        "wrong_problem",
        "mixed",
        "unknown",
        "other",
    ]
    return [v for v in preferred if v in VALID_CRITIQUE_DEBATE_VERDICTS or v == "other"]


def hash_seed(username: str) -> int:
    return int(hashlib.sha256(username.encode("utf-8")).hexdigest(), 16) % (2**32)


def deterministic_order(tasks: List[JudgingTask], seed: int) -> List[JudgingTask]:
    ordered = sorted(tasks, key=lambda t: t.id or "")
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def deterministic_order_keys(entries: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    ordered = sorted(entries, key=lambda e: e.get("id") or "")
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered


def max_other_confidence(other_labels: List[HumanEvaluation]) -> int:
    confidences = [
        label.confidence or 0
        for label in other_labels
        if label and (label.verdict or "").lower() != "unknown"
    ]
    return max(confidences) if confidences else 0


def normalize_key(raw: Optional[List[Any]]) -> Optional[Tuple[Optional[str], Optional[str], Optional[int]]]:
    if not raw or len(raw) != 3:
        return None
    run_id, topic_slug, idx = raw
    try:
        idx_val = int(idx) if idx is not None else None
    except (TypeError, ValueError):
        idx_val = None
    return (str(run_id) if run_id is not None else None, topic_slug, idx_val)


def load_key_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    entries: List[Dict[str, Any]] = []
    for section in ("illposed", "critiques"):
        for item in data.get(section, []):
            if not item:
                continue
            entries.append(item)
    return entries


def collect_consensus(auto_eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    consensus: Dict[str, Dict[str, Any]] = {}
    if not auto_eval_dir.exists():
        return consensus
    for eval_file in auto_eval_dir.glob("*.json"):
        payload = load_evaluation_entries(eval_file)
        for decision in payload.decisions:
            if not decision.id or decision.verdict is None:
                continue
            bucket = consensus.setdefault(decision.id, {"verdicts": []})
            bucket["verdicts"].append(decision.verdict)
    for key, bucket in consensus.items():
        verdicts = bucket["verdicts"]
        bucket["consensus"] = bool(verdicts) and len(set(verdicts)) == 1
        bucket["verdict"] = verdicts[-1] if verdicts else None
        bucket["count"] = len(verdicts)
    return consensus


def collect_auto_evals(auto_eval_dir: Path) -> Tuple[Dict[str, List[AutomatedEvaluation]], Dict[Tuple, List[AutomatedEvaluation]]]:
    by_id: Dict[str, List[AutomatedEvaluation]] = {}
    by_key: Dict[Tuple, List[AutomatedEvaluation]] = {}
    if not auto_eval_dir.exists():
        return by_id, by_key
    for eval_file in auto_eval_dir.glob("*.json"):
        payload = load_evaluation_entries(eval_file)
        for decision in payload.decisions:
            if not decision.id:
                continue
            by_id.setdefault(decision.id, []).append(decision)
            comp = (
                decision.type,
                decision.mode,
                decision.question_model,
                decision.answer_model,
                decision.critic_model,
                str(decision.run_id) if decision.run_id is not None else None,
            )
            by_key.setdefault(comp, []).append(decision)
    return by_id, by_key


KeyTuple = Tuple[Optional[str], Optional[str], Optional[int]]


def build_answer_cache(answers_dir: Path) -> Dict[Tuple[str, str], Dict[str, AnswerEntry]]:
    cache: Dict[Tuple[str, str], Dict[str, AnswerEntry]] = {}
    for q_dir in answers_dir.glob("*"):
        q_slug = q_dir.name
        for ans_file in q_dir.glob("*.json"):
            a_slug = ans_file.stem
            records = load_answer_entries(ans_file)
            mapping: Dict[str, AnswerEntry] = {}
            for entry in records:
                if not entry or entry.run_id is None:
                    continue
                key = str(entry.run_id)
                if key not in mapping:
                    mapping[key] = entry
            cache[(q_slug, a_slug)] = mapping
    return cache


def build_benchmark_cache(benchmark_dir: Path) -> Dict[str, Dict[str, AnswerEntry]]:
    cache: Dict[str, Dict[str, AnswerEntry]] = {}
    for bench_path in benchmark_dir.glob("*.json"):
        q_slug = bench_path.stem
        entries = load_benchmark_entries(bench_path)
        answers = benchmark_answers_from_entries(q_slug, entries)
        mapping: Dict[str, AnswerEntry] = {}
        for entry in answers:
            if not entry or entry.run_id is None:
                continue
            key = str(entry.run_id)
            if key not in mapping:
                mapping[key] = entry
        cache[q_slug] = mapping
    return cache


def build_debate_cache_illposed(debates_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for debate_file in debates_dir.glob("illposed/*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        entries = load_debate_entries(debate_file)
        mapping: Dict[str, Any] = {}
        for entry in entries:
            if not entry or entry.run_id is None:
                continue
            key = str(entry.run_id)
            if key not in mapping:
                mapping[key] = entry
        cache[(q_slug, a_slug)] = mapping
    return cache


def build_debate_cache_critiques(debates_dir: Path) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    cache: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    critiques_root = debates_dir / "critiques"
    if not critiques_root.exists():
        return cache
    for mode_dir in critiques_root.glob("*"):
        mode = mode_dir.name
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            for debate_file in q_dir.glob("*.json"):
                parts = debate_file.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                entries = load_debate_entries(debate_file)
                mapping: Dict[str, Any] = {}
                for entry in entries:
                    if not entry or entry.run_id is None:
                        continue
                    key = str(entry.run_id)
                    if key not in mapping:
                        mapping[key] = entry
                cache[(mode, q_slug, critic_slug, answer_slug)] = mapping
    return cache


def build_critique_cache(critiques_dir: Path) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    cache: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for mode_dir in critiques_dir.glob("*"):
        mode = mode_dir.name
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__")
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                entries = load_critique_entries(crit_file)
                mapping: Dict[str, Any] = {}
                for entry in entries:
                    if not entry or entry.run_id is None:
                        continue
                    key = str(entry.run_id)
                    if key not in mapping:
                        mapping[key] = entry
                cache[(mode, q_slug, critic_slug, answer_slug)] = mapping
    return cache


def _lookup_run(mapping: Dict[str, Any], run_id: Optional[str]) -> Any:
    if run_id is None:
        return None
    return mapping.get(str(run_id))


def build_task_from_key(
    key_entry: Dict[str, Any],
    registry,
    answer_cache,
    benchmark_cache,
    debate_cache_illposed,
    debate_cache_critiques,
    critique_cache,
) -> Tuple[Optional[JudgingTask], bool]:
    key_tuple = normalize_key(key_entry.get("key"))
    if not key_tuple:
        return None, False
    run_id, topic_slug, idx = key_tuple
    k_type = key_entry.get("type")
    if k_type == "illposed":
        q_slug = key_entry.get("question_model")
        a_slug = key_entry.get("answer_model")
        if not q_slug or not a_slug:
            return None, False
        answers = answer_cache.get((q_slug, a_slug), {})
        debates = debate_cache_illposed.get((q_slug, a_slug), {})
        debate = _lookup_run(debates, run_id)
        answer_entry = _lookup_run(answers, run_id)
        fallback_answers = benchmark_cache.get(q_slug, {})
        fallback_entry = _lookup_run(fallback_answers, run_id)
        question_text = (debate.question if debate else "") or (answer_entry.question if answer_entry else key_entry.get("question") or "") or (fallback_entry.question if fallback_entry else "")
        answer_text = final_answer(answer_entry) if answer_entry else ""
        if not answer_text and fallback_entry:
            answer_text = final_answer(fallback_entry) or ""
        history = debate.history if debate else []
        has_data = bool(history)
        claim_text = build_illposed_claim(answer_entry) or build_illposed_claim(fallback_entry)
        return (
            JudgingTask(
                id=key_entry.get("id") or _task_id(f"illposed/{q_slug}/{a_slug}", run_id, topic_slug, question_text or ""),
                type="illposed",
                question_model=q_slug,
                answer_model=a_slug,
                critic_model=a_slug,
                run_id=run_id,
                topic_slug=topic_slug,
                question=question_text,
                answer=answer_text,
                critique=claim_text,
                debate_history=history,
            ),
            has_data,
        )
    if k_type == "critique":
        mode = key_entry.get("mode")
        q_slug = key_entry.get("question_model")
        critic_slug = key_entry.get("critic_model")
        answer_slug = key_entry.get("answer_model")
        if not mode or not q_slug or not critic_slug or not answer_slug:
            return None, False

        answers = answer_cache.get((q_slug, answer_slug), {})
        answer_entry = _lookup_run(answers, run_id)
        fallback_answers = benchmark_cache.get(q_slug, {})
        fallback_entry = _lookup_run(fallback_answers, run_id)

        debates = debate_cache_critiques.get((mode, q_slug, critic_slug, answer_slug), {})
        critiques = critique_cache.get((mode, q_slug, critic_slug, answer_slug), {})
        debate = _lookup_run(debates, run_id)
        critique_entry = _lookup_run(critiques, run_id)

        if not debate or not debate.history:
            return None, False

        question_text = (debate.question if debate else "") or (critique_entry.question if critique_entry else key_entry.get("question") or "") or (fallback_entry.question if fallback_entry else "")
        answer_text = final_answer(answer_entry) if answer_entry else ""
        if not answer_text and fallback_entry:
            answer_text = final_answer(fallback_entry) or ""
        attempts = critique_entry.attempts if critique_entry else None
        last_attempt = attempts[-1] if attempts else None
        critique_text = (
            (last_attempt.notes or "").strip()
            or (last_attempt.cleaned_critique or "").strip()
            or (last_attempt.raw_critique or "").strip()
            if last_attempt
            else ""
        )
        history = debate.history if debate else []
        has_data = bool(history)
        display_critic_slug = critic_slug
        return (
            JudgingTask(
                id=key_entry.get("id") or _task_id(
                    f"critique/{mode}/{q_slug}/{display_critic_slug}__{answer_slug}",
                    run_id,
                    topic_slug,
                    question_text or "",
                ),
                type="critique",
                mode=mode,
                question_model=q_slug,
                answer_model=answer_slug,
                critic_model=display_critic_slug,
                run_id=run_id,
                topic_slug=topic_slug,
                question=question_text,
                answer=answer_text,
                critique=critique_text,
                debate_history=history,
            ),
            has_data,
        )
    return None, False


def render_task(
    task: JudgingTask,
    existing: Optional[HumanEvaluation],
    other_labels: List[HumanEvaluation],
    auto_evals: List[AutomatedEvaluation],
    registry,
    save_cb,
):
    st.markdown(f"### {task.id}")
    alice = task.critic_model if task.type == "critique" else task.answer_model
    bob = task.answer_model if task.type == "critique" else task.question_model
    display_q = registry.display_name_for_slug(task.question_model or "") if registry else task.question_model
    display_a = registry.display_name_for_slug(task.answer_model or "") if registry else task.answer_model
    display_c = registry.display_name_for_slug(task.critic_model or "") if registry else task.critic_model

    st.write(
        f"**Question model:** {display_q}  |  **Answer model:** {display_a}  |  **Critic:** {display_c or ''}"
    )
    st.info(f"Alice: {display_c or display_a or 'unknown'}   |   Bob: {display_a if task.type == 'critique' else display_q or 'unknown'}")
    question_text = task.question or ""
    answer_text = task.answer or ""
    critique_text = task.critique or ""
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Question")
        st.write(question_text)
        st.download_button("Copy Question", data=question_text, file_name="Question.md", mime="text/markdown")
        st.subheader("Answer")
        st.write(answer_text)
        st.download_button("Copy Answer", data=answer_text, file_name="Answer.md", mime="text/markdown")
    with cols[1]:
        if task.type in ("critique", "illposed"):
            st.subheader("Critique")
            st.markdown(critique_text)
            st.download_button("Copy Critique", data=critique_text, file_name="Critique.md", mime="text/markdown")
        st.subheader("Debate")
        for msg in task.debate_history or []:
            st.markdown(f"- **{msg.speaker}**: {msg.message}")

    if other_labels:
        summary = ", ".join(
            f"{lbl.verdict or 'unknown'} (conf {lbl.confidence or '?'})"
            for lbl in other_labels
            if lbl
        )
        st.info(f"Other evaluations recorded: {summary}")
    if auto_evals:
        st.subheader("Automated evaluations")
        for ev in auto_evals:
            st.markdown(
                f"- Judge `{ev.judge_model or 'unknown'}` -> **{ev.verdict or 'unknown'}** (conf {ev.confidence or '?'}): {ev.reasoning or ''}"
            )

    st.divider()
    if auto_evals:
        combined = "\n\n".join(
            f"Judge: {ev.judge_model or 'unknown'}\nVerdict: {ev.verdict}\nConfidence: {ev.confidence}\nReasoning: {ev.reasoning or ''}"
            for ev in auto_evals
        )
        st.download_button("Copy automated evaluations", data=combined, file_name="automated_evaluations.md", mime="text/markdown")

    all_fields = []
    all_fields.append(f"Question:\n{task.question or ''}")
    all_fields.append(f"Answer:\n{task.answer or ''}")
    if task.type in ("critique", "illposed"):
        all_fields.append(f"Critique:\n{task.critique or ''}")
    if auto_evals:
        all_fields.append("Automated evaluations:")
        all_fields.extend(
            [
                f"- {ev.judge_model or 'unknown'}: {ev.verdict} (conf {ev.confidence}) {ev.reasoning or ''}"
                for ev in auto_evals
            ]
        )
    st.download_button("Copy all fields", data="\n\n".join(all_fields), file_name="task.md", mime="text/markdown")

    options = verdict_choices(task.type)
    existing_choice = existing.verdict if existing else None
    default_index = options.index(existing_choice) if existing_choice in options else options.index("unknown")
    choice = st.radio("Verdict", options, index=default_index, key=f"verdict-{task.id}")
    confidence = st.slider(
        "Confidence (1-5)",
        1,
        5,
        int(existing.confidence if existing and existing.confidence is not None else 3),
        1,
        key=f"conf-{task.id}",
    )
    comment = st.text_area("Comments", value=existing.comment if existing else "", key=f"comment-{task.id}")
    if st.button("Save", key=f"save-{task.id}"):
        ok, message, output = save_cb(
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
        if ok:
            st.success(message)
        else:
            st.error(message)
        if output and not ok:
            st.code(output)


def main():
    parser = argparse.ArgumentParser(description="Ordered interactive web labeller")
    parser.add_argument("--username", required=True)
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--evaluations-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--automated-evals-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--keys", type=Path, default=Path("debate_keys.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    args, _ = parser.parse_known_args()

    registry = load_registry(str(args.config))

    key_entries = load_key_entries(args.keys)
    consensus = collect_consensus(args.automated_evals_dir)
    auto_eval_by_id, auto_eval_by_key = collect_auto_evals(args.automated_evals_dir)

    answer_cache = build_answer_cache(args.answers_dir)
    benchmark_cache = build_benchmark_cache(args.benchmark_dir)
    debate_cache_illposed = build_debate_cache_illposed(args.debates_dir)
    debate_cache_critiques = build_debate_cache_critiques(args.debates_dir)
    critique_cache = build_critique_cache(args.critiques_dir)

    seed = hash_seed(args.username)
    ordered_keys = deterministic_order_keys(key_entries, seed)

    tasks: List[Tuple[JudgingTask, List[AutomatedEvaluation]]] = []
    dropped_no_data = 0
    dropped_no_evals = 0
    dropped_unanimous = 0
    for entry in ordered_keys:
        task, has_data = build_task_from_key(
            entry,
            registry,
            answer_cache,
            benchmark_cache,
            debate_cache_illposed,
            debate_cache_critiques,
            critique_cache,
        )
        if not has_data:
            dropped_no_data += 1
            continue
        if not task:
            continue
        comp_key = (
            task.type,
            task.mode,
            task.question_model,
            task.answer_model,
            task.critic_model,
            str(task.run_id) if task.run_id is not None else None,
        )
        evals = auto_eval_by_id.get(task.id, []) or auto_eval_by_key.get(comp_key, [])
        if not evals:
            dropped_no_evals += 1
            continue
        verdicts = {ev.verdict for ev in evals if ev and ev.verdict}
        if verdicts and len(verdicts) == 1:
            dropped_unanimous += 1
            continue
        tasks.append((task, evals))

    eval_path = args.evaluations_dir / f"{args.username}.json"
    decisions = load_evaluations(eval_path)
    other_decisions = load_other_evaluations(args.evaluations_dir, args.username)

    st.title(f"Labeller (ordered) - {args.username}")

    claim_filter = st.multiselect(
        "Claim type",
        ["illposed", "critique"],
        default=["illposed", "critique"],
    )
    questioners = sorted({t.question_model for t, _ in tasks if t.question_model})
    answerers = sorted({t.answer_model for t, _ in tasks if t.answer_model})
    critics = sorted({t.critic_model for t, _ in tasks if t.critic_model})
    selected_q = st.multiselect("Question model", questioners, default=questioners)
    selected_a = st.multiselect("Answer model", answerers, default=answerers)
    selected_c = st.multiselect("Critic", critics, default=critics)
    only_unlabeled = st.checkbox("Only not yet labelled by me", value=True)
    include_other_labels = st.checkbox("Include tasks labelled by others (confidence ≥ 3)", value=False)
    skip_labeled_by_others = not include_other_labels

    filtered = []
    skipped_by_others = 0
    for task, evals in tasks:
        if task.type not in claim_filter:
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
        other_labels = other_decisions.get(task.id, [])
        if skip_labeled_by_others and max_other_confidence(other_labels) >= 3:
            skipped_by_others += 1
            continue
        filtered.append((task, evals, other_labels))

    ordered_pairs = filtered

    st.subheader(f"Tasks ({len(ordered_pairs)})")
    if skip_labeled_by_others and skipped_by_others:
        st.info(
            f"Skipped {skipped_by_others} tasks already labelled elsewhere with confidence ≥ 3 (excluding 'unknown'). "
            "Tick the checkbox above to include them."
        )
    if dropped_no_data:
        st.info(f"Dropped {dropped_no_data} keys lacking debate data.")
    if dropped_unanimous:
        st.info(f"Skipped {dropped_unanimous} unanimously judged tasks.")

    if "task_idx" not in st.session_state:
        st.session_state["task_idx"] = 0

    if not ordered_pairs:
        st.info("No tasks match the current filters.")
        return

    idx = st.session_state.get("task_idx", 0)
    if idx >= len(ordered_pairs):
        idx = len(ordered_pairs) - 1
        st.session_state["task_idx"] = idx
    task, evals, other_labels = ordered_pairs[idx]
    existing = decisions.get(task.id)

    st.text(f"Task {idx + 1} / {len(ordered_pairs)}")
    nav_cols = st.columns(3)
    with nav_cols[0]:
        if st.button("Previous", disabled=idx <= 0):
            st.session_state["task_idx"] = max(idx - 1, 0)
            st.rerun()
    with nav_cols[1]:
        if st.button("Next", disabled=idx >= len(ordered_pairs) - 1):
            st.session_state["task_idx"] = min(idx + 1, len(ordered_pairs) - 1)
            st.rerun()
    with nav_cols[2]:
        jump = st.number_input("Jump to", min_value=1, max_value=len(ordered_pairs), value=idx + 1, step=1)
        if jump - 1 != idx:
            st.session_state["task_idx"] = int(jump - 1)
            st.rerun()

    def save_decision(decision: HumanEvaluation) -> Tuple[bool, str, str]:
        decisions[decision.id] = decision
        save_evaluation(eval_path, decisions)
        label_id = decision.id or "unknown"
        return run_git_sync(label_id, args.username, eval_path)

    st.divider()
    render_task(task, existing, other_labels, evals, registry, save_decision)

    with st.expander("Guidance - Question"):
        st.markdown(load_question_guidance())
    with st.expander("Guidance - Answer"):
        st.markdown(load_answer_guidance())
    with st.expander("Guidance - Critique"):
        st.markdown(load_critique_guidance())
    with st.expander("Guidance - Evaluation"):
        eval_guidance = load_judgment_critique_guidance() if task.type == "critique" else load_judgment_illposed_guidance()
        st.markdown(eval_guidance)


if __name__ == "__main__":
    main()
