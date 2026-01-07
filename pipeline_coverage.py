import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from constants import (
    CRITIQUE_VERDICT_CORRECT,
    CRITIQUE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
)
from data_models import (
    AnswerEntry,
    BenchmarkEntry,
    CritiqueEntry,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_evaluation_entries,
)
from model_config import ModelSpec, _slugify, load_registry
from utils import benchmark_answers_from_entries


def _load_runs(path: Path, limit: Optional[int]) -> List[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        run_ids = [str(k) for k in data.keys()]
    elif isinstance(data, list):
        run_ids = []
        for idx, item in enumerate(data):
            if isinstance(item, dict) and "run_id" in item:
                run_ids.append(str(item["run_id"]))
            else:
                run_ids.append(str(idx))
    else:
        raise ValueError(f"Unsupported runs file format: {type(data).__name__}")
    if limit is not None:
        run_ids = run_ids[:limit]
    return run_ids


def _entries_by_run(entries: Sequence[Optional[Union[BenchmarkEntry, AnswerEntry]]]) -> Dict[str, object]:
    mapping: Dict[str, object] = {}
    for entry in entries:
        if entry and getattr(entry, "run_id", None) is not None:
            mapping[str(entry.run_id)] = entry
    return mapping


def _pick_entry(entries: Sequence[Optional[Union[BenchmarkEntry, AnswerEntry]]], entries_by_run: Dict[str, object], run_idx: int, run_id: str):
    entry = entries_by_run.get(run_id)
    if entry is not None:
        return entry
    if run_idx < len(entries):
        return entries[run_idx]
    return None


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return f"{numerator}/0 (n/a)"
    pct = 100.0 * numerator / denominator
    return f"{numerator}/{denominator} ({pct:.1f}%)"


def _final_question(entry: Optional[BenchmarkEntry]) -> Optional[str]:
    if not entry:
        return None
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question


def _final_critique_verdict(entry: Optional[CritiqueEntry]) -> Optional[str]:
    if not entry:
        return None
    attempts = entry.attempts or []
    if not attempts:
        return None
    return attempts[-1].verdict


def _has_critique_text(entry: Optional[CritiqueEntry]) -> bool:
    if not entry:
        return False
    attempts = entry.attempts or []
    if not attempts:
        return False
    last = attempts[-1]
    if not last.verdict:
        return False
    if last.notes is None or not isinstance(last.notes, str):
        return False
    return True


def _load_answer_records(
    answers_dir: Path,
    q_slug: str,
    answer_slug: str,
    benchmark_entries: Sequence[Optional[BenchmarkEntry]],
) -> List[Optional[AnswerEntry]]:
    answer_path = answers_dir / q_slug / f"{answer_slug}.json"
    if answer_path.exists():
        return load_answer_entries(answer_path)
    if answer_slug == q_slug:
        return benchmark_answers_from_entries(q_slug, list(benchmark_entries))
    return []


def _count_question_stats(
    benchmark_dir: Path,
    benchmark_specs: Sequence[ModelSpec],
    run_ids: Sequence[str],
) -> Tuple[Counter, int, Dict[str, Set[str]]]:
    counts: Counter = Counter()
    success_runs_by_slug: Dict[str, Set[str]] = {}
    for spec in benchmark_specs:
        q_slug = spec.slug
        entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
        entries_by_run = _entries_by_run(entries)
        success_runs: Set[str] = set()
        for run_idx, run_id in enumerate(run_ids):
            entry = _pick_entry(entries, entries_by_run, run_idx, run_id)
            if entry:
                counts["generated"] += 1
                if entry.status == STATUS_SUCCEEDED:
                    counts["succeeded"] += 1
                    success_runs.add(run_id)
                elif entry.status == STATUS_ILL_POSED:
                    counts["ill_posed"] += 1
                elif entry.status == STATUS_FAILED:
                    counts["failed"] += 1
            else:
                counts["missing"] += 1
        success_runs_by_slug[q_slug] = success_runs
    expected = len(run_ids) * len(benchmark_specs)
    return counts, expected, success_runs_by_slug


def _count_answer_stats(
    answers_dir: Path,
    answer_specs: Sequence[ModelSpec],
    benchmark_specs: Sequence[ModelSpec],
    run_ids: Sequence[str],
    success_runs_by_slug: Dict[str, Set[str]],
    allow_self_answering: bool,
) -> Tuple[Counter, int]:
    counts: Counter = Counter()
    expected = 0
    for q_spec in benchmark_specs:
        q_slug = q_spec.slug
        success_runs = success_runs_by_slug.get(q_slug, set())
        eligible_answers = [
            spec for spec in answer_specs
            if allow_self_answering or spec.slug != q_slug
        ]
        expected += len(success_runs) * len(eligible_answers)
        for answer_spec in eligible_answers:
            answer_path = answers_dir / q_slug / f"{answer_spec.slug}.json"
            entries = load_answer_entries(answer_path) if answer_path.exists() else []
            entries_by_run = _entries_by_run(entries)
            for run_idx, run_id in enumerate(run_ids):
                if run_id not in success_runs:
                    continue
                entry = _pick_entry(entries, entries_by_run, run_idx, run_id)
                if entry:
                    counts["generated"] += 1
                    if entry.status == STATUS_SUCCEEDED:
                        counts["succeeded"] += 1
                    elif entry.status == STATUS_ILL_POSED:
                        counts["ill_posed"] += 1
                    elif entry.status == STATUS_FAILED:
                        counts["failed"] += 1
                else:
                    counts["missing"] += 1
    return counts, expected


def _expected_critiques_for_question(
    mode: str,
    question_model: ModelSpec,
    benchmark_entries: Sequence[Optional[BenchmarkEntry]],
    answer_specs: Sequence[ModelSpec],
    critic_names: Set[str],
    answers_dir: Path,
    registry,
    limit: Optional[int],
    custom_pairs: Sequence[Tuple[str, str, str]],
) -> int:
    q_slug = question_model.slug
    question_author_answers = _load_answer_records(
        answers_dir,
        q_slug,
        q_slug,
        benchmark_entries,
    )

    def has_budget(current: int) -> bool:
        return limit is None or current < limit

    count = 0
    if mode == "custom":
        if not custom_pairs:
            return 0
        for question_author, answer_author, critic in custom_pairs:
            if question_author != question_model.name:
                continue
            if critic_names and critic not in critic_names:
                continue
            critic_spec = registry.models.get(critic)
            if not critic_spec or "critique" not in critic_spec.roles:
                continue
            answer_slug = _slugify(answer_author)
            answer_records = _load_answer_records(
                answers_dir,
                q_slug,
                answer_slug,
                benchmark_entries,
            )
            for idx, answer_entry in enumerate(answer_records):
                if not has_budget(count):
                    return count
                question_entry = benchmark_entries[idx] if idx < len(benchmark_entries) else None
                if not question_entry or not _final_question(question_entry):
                    continue
                if not answer_entry or answer_entry.status == STATUS_FAILED:
                    continue
                count += 1
        return count

    if mode == "contradictor":
        critic_list = sorted(critic_names) if critic_names else [spec.name for spec in answer_specs]
        for critic_model in critic_list:
            if critic_model == question_model.name:
                continue
            critic_spec = registry.models.get(critic_model)
            if not critic_spec or "critique" not in critic_spec.roles:
                continue
            for idx, answer_entry in enumerate(question_author_answers):
                if not has_budget(count):
                    return count
                if not answer_entry or answer_entry.status == STATUS_FAILED:
                    continue
                question_entry = benchmark_entries[idx] if idx < len(benchmark_entries) else None
                if not question_entry or not _final_question(question_entry):
                    continue
                count += 1
        return count

    if mode == "evaluator" and critic_names and question_model.name not in critic_names:
        return 0

    for answer_spec in answer_specs:
        if mode in {"all", "custom"} and critic_names and answer_spec.name not in critic_names:
            continue
        answer_records = _load_answer_records(
            answers_dir,
            q_slug,
            answer_spec.slug,
            benchmark_entries,
        )
        if answer_spec.name == question_model.name and not answer_records:
            answer_records = question_author_answers

        for idx, answer_entry in enumerate(answer_records):
            if not has_budget(count):
                return count
            question_author_entry = question_author_answers[idx] if idx < len(question_author_answers) else None
            if not question_author_entry or not answer_entry:
                continue
            if mode == "evaluator" and answer_spec.name == question_model.name:
                continue
            if answer_entry.status == STATUS_FAILED:
                continue
            question_entry = benchmark_entries[idx] if idx < len(benchmark_entries) else None
            if not question_entry or not _final_question(question_entry):
                continue
            count += 1

    return count


def _count_critique_stats(
    critiques_dir: Path,
    benchmark_dir: Path,
    benchmark_specs: Sequence[ModelSpec],
    answer_specs: Sequence[ModelSpec],
    critic_specs: Sequence[ModelSpec],
    critique_modes: Sequence[str],
    answers_dir: Path,
    registry,
    limit: Optional[int],
    custom_pairs: Sequence[Tuple[str, str, str]],
) -> Tuple[Dict[str, Counter], Dict[str, int]]:
    allowed_slugs = {spec.slug for spec in benchmark_specs}
    expected_by_mode: Dict[str, int] = {}
    actual_by_mode: Dict[str, Counter] = {}
    critic_names = {spec.name for spec in critic_specs}

    for mode in critique_modes:
        expected = 0
        for spec in benchmark_specs:
            benchmark_entries = load_benchmark_entries(benchmark_dir / f"{spec.slug}.json")
            expected += _expected_critiques_for_question(
                mode,
                spec,
                benchmark_entries,
                answer_specs,
                critic_names,
                answers_dir,
                registry,
                limit,
                custom_pairs,
            )
        expected_by_mode[mode] = expected

        counts: Counter = Counter()
        mode_dir = critiques_dir / mode
        if mode_dir.exists():
            for q_dir in mode_dir.glob("*"):
                if q_dir.name not in allowed_slugs:
                    continue
                for crit_file in q_dir.glob("*.json"):
                    entries = load_critique_entries(crit_file)
                    for entry in entries:
                        if entry:
                            counts["generated"] += 1
                            if entry.status == STATUS_SUCCEEDED:
                                counts["succeeded"] += 1
                            else:
                                counts["failed"] += 1
                        else:
                            counts["missing"] += 1
        actual_by_mode[mode] = counts

    return actual_by_mode, expected_by_mode


def _count_debate_stats(
    debates_dir: Path,
    critiques_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    critique_modes: Sequence[str],
    benchmark_specs: Sequence[ModelSpec],
) -> Tuple[Counter, Counter, Dict[str, int], Dict[str, int]]:
    actual_illposed = Counter()
    actual_critique_by_mode: Counter = Counter()
    expected_illposed = Counter()
    expected_critique_by_mode: Dict[str, int] = {}
    allowed_slugs = {spec.slug for spec in benchmark_specs}

    for q_dir in answers_dir.glob("*"):
        q_slug = q_dir.name
        if allowed_slugs and q_slug not in allowed_slugs:
            continue
        for answer_file in q_dir.glob("*.json"):
            entries = load_answer_entries(answer_file)
            expected_illposed["expected"] += sum(
                1 for entry in entries if entry and entry.status == STATUS_ILL_POSED
            )

    for debate_file in (debates_dir / "illposed").glob("*/*.json"):
        entries = load_debate_entries(debate_file)
        actual_illposed["generated"] += sum(1 for entry in entries if entry)

    for mode in critique_modes:
        expected = 0
        mode_dir = critiques_dir / mode
        if mode_dir.exists():
            for q_dir in mode_dir.glob("*"):
                q_slug = q_dir.name
                if allowed_slugs and q_slug not in allowed_slugs:
                    continue
                benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
                for crit_file in q_dir.glob("*.json"):
                    parts = crit_file.stem.split("__")
                    if len(parts) != 2:
                        continue
                    _critic_slug, answer_slug = parts
                    answers = _load_answer_records(
                        answers_dir,
                        q_slug,
                        answer_slug,
                        benchmark_entries,
                    )
                    critiques = load_critique_entries(crit_file)
                    for idx, crit_entry in enumerate(critiques):
                        if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                            continue
                        verdict = _final_critique_verdict(crit_entry)
                        if verdict in {None, CRITIQUE_VERDICT_UNKNOWN, CRITIQUE_VERDICT_CORRECT}:
                            continue
                        if idx >= len(answers) or not answers[idx]:
                            continue
                        expected += 1
        expected_critique_by_mode[mode] = expected

        debate_mode_dir = debates_dir / "critiques" / mode
        if debate_mode_dir.exists():
            for q_dir in debate_mode_dir.glob("*"):
                if q_dir.name not in allowed_slugs:
                    continue
                for debate_file in q_dir.glob("*.json"):
                    entries = load_debate_entries(debate_file)
                    actual_critique_by_mode[mode] += sum(1 for entry in entries if entry)

    return actual_illposed, actual_critique_by_mode, expected_illposed, expected_critique_by_mode


def _collect_judging_tasks(
    debates_dir: Path,
    critiques_dir: Path,
    answers_dir: Path,
    benchmark_dir: Path,
    registry,
    critique_modes: Sequence[str],
    allow_no_debate: bool,
    force_correct_critiques: bool,
) -> Dict[str, Tuple[str, str]]:
    tasks: Dict[str, Tuple[str, str]] = {}

    for debate_file in (debates_dir / "illposed").glob("*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        benchmark_entries = load_benchmark_entries(benchmark_dir / f"{q_slug}.json")
        answers = _load_answer_records(
            answers_dir,
            q_slug,
            a_slug,
            benchmark_entries,
        )
        debates = load_debate_entries(debate_file)
        max_len = max(len(debates), len(answers))
        for idx in range(max_len):
            debate = debates[idx] if idx < len(debates) else None
            answer_entry = answers[idx] if idx < len(answers) else None
            status = answer_entry.status if answer_entry else None
            if status and status != STATUS_ILL_POSED:
                continue
            history = debate.history if debate else []
            if not history and not allow_no_debate:
                continue
            if not debate and not answer_entry:
                continue
            alice_model = registry.resolve_model_name((debate.alice_model if debate else None) or a_slug)
            bob_model = registry.resolve_model_name((debate.bob_model if debate else None) or q_slug)
            task_id = f"illposed/{q_slug}/{a_slug}/{idx}"
            tasks[task_id] = (alice_model, bob_model)

    for mode in critique_modes:
        mode_dir = critiques_dir / mode
        if not mode_dir.exists():
            continue
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
                for idx, crit_entry in enumerate(critiques):
                    if not crit_entry or crit_entry.status != STATUS_SUCCEEDED:
                        continue
                    verdict = _final_critique_verdict(crit_entry)
                    if verdict in {None, CRITIQUE_VERDICT_UNKNOWN}:
                        continue
                    if verdict == CRITIQUE_VERDICT_CORRECT and not force_correct_critiques:
                        continue
                    debate = debates[idx] if idx < len(debates) else None
                    history = debate.history if debate else []
                    if not history and not allow_no_debate:
                        continue
                    if not _has_critique_text(crit_entry):
                        continue
                    alice_model = registry.resolve_model_name((debate.alice_model if debate else None) or critic_slug)
                    bob_model = registry.resolve_model_name((debate.bob_model if debate else None) or answer_slug)
                    task_id = f"critique/{mode}/{q_slug}/{critic_slug}__{answer_slug}/{idx}"
                    tasks[task_id] = (alice_model, bob_model)

    return tasks


def _count_automated_evaluations(
    automated_dir: Path,
    tasks: Dict[str, Tuple[str, str]],
    judge_specs: Sequence[ModelSpec],
) -> Tuple[int, int]:
    judge_names = [spec.name for spec in judge_specs]
    expected = 0
    for alice_model, bob_model in tasks.values():
        participants = {alice_model, bob_model}
        expected += sum(1 for judge in judge_names if judge not in participants)

    actual = 0
    task_ids = set(tasks.keys())
    for eval_file in automated_dir.glob("*.json"):
        payload = load_evaluation_entries(eval_file)
        for decision in payload.decisions:
            if decision.id in task_ids:
                actual += 1
    return actual, expected


def _parse_custom_map(path: Optional[Path]) -> List[Tuple[str, str, str]]:
    if not path:
        return []
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    pairs = []
    for item in payload:
        question_author = item.get("question_author") or item.get("question_owner")
        answer_author = item.get("answer_author") or item.get("answerer")
        critic = item.get("critic")
        if not question_author or not answer_author or not critic:
            continue
        pairs.append((question_author, answer_author, critic))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute pipeline coverage stats with percentages.")
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--runs-file", type=Path, default=Path("configs/runs.json"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--automated-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-self-answering", action="store_true")
    parser.add_argument("--allow-no-debate", action="store_true")
    parser.add_argument("--force-correct-critiques", action="store_true")
    parser.add_argument("--benchmark-models", nargs="*")
    parser.add_argument("--answer-models", nargs="*")
    parser.add_argument("--critique-models", nargs="*")
    parser.add_argument("--judge-models", nargs="*")
    parser.add_argument("--critique-modes", nargs="*", default=["contradictor", "evaluator"])
    parser.add_argument("--custom-map", type=Path)
    args = parser.parse_args()

    registry = load_registry(str(args.config))
    run_ids = _load_runs(args.runs_file, args.limit)

    benchmark_specs = registry.pick(args.benchmark_models) if args.benchmark_models else registry.by_role("benchmark")
    answer_specs = registry.pick(args.answer_models) if args.answer_models else registry.by_role("answer")
    critic_specs = registry.pick(args.critique_models) if args.critique_models else registry.by_role("critique")
    judge_specs = registry.pick(args.judge_models) if args.judge_models else list(registry.models.values())

    custom_pairs = _parse_custom_map(args.custom_map)

    question_counts, question_expected, success_runs_by_slug = _count_question_stats(
        args.benchmark_dir,
        benchmark_specs,
        run_ids,
    )
    answer_counts, answer_expected = _count_answer_stats(
        args.answers_dir,
        answer_specs,
        benchmark_specs,
        run_ids,
        success_runs_by_slug,
        args.allow_self_answering,
    )
    critique_counts_by_mode, critique_expected_by_mode = _count_critique_stats(
        args.critiques_dir,
        args.benchmark_dir,
        benchmark_specs,
        answer_specs,
        critic_specs,
        args.critique_modes,
        args.answers_dir,
        registry,
        args.limit,
        custom_pairs,
    )
    actual_illposed, actual_critique_by_mode, expected_illposed, expected_critique_by_mode = _count_debate_stats(
        args.debates_dir,
        args.critiques_dir,
        args.answers_dir,
        args.benchmark_dir,
        args.critique_modes,
        benchmark_specs,
    )
    judging_tasks = _collect_judging_tasks(
        args.debates_dir,
        args.critiques_dir,
        args.answers_dir,
        args.benchmark_dir,
        registry,
        args.critique_modes,
        args.allow_no_debate,
        args.force_correct_critiques,
    )
    eval_actual, eval_expected = _count_automated_evaluations(
        args.automated_dir,
        judging_tasks,
        judge_specs,
    )

    print("Pipeline coverage:")
    print(f"- Runs: {len(run_ids)}")
    print(f"- Benchmark models: {len(benchmark_specs)}")
    print(f"- Answer models: {len(answer_specs)}")
    print(f"- Critique models: {len(critic_specs)}")
    print(f"- Judge models: {len(judge_specs)}")

    print("\nQuestions:")
    print(f"- Generated: {_format_ratio(question_counts['generated'], question_expected)}")
    if question_counts["succeeded"]:
        print(f"- Succeeded: {_format_ratio(question_counts['succeeded'], question_expected)}")
    if question_counts["ill_posed"]:
        print(f"- Ill-posed: {_format_ratio(question_counts['ill_posed'], question_expected)}")
    if question_counts["failed"]:
        print(f"- Failed: {_format_ratio(question_counts['failed'], question_expected)}")

    print("\nAnswers:")
    print(f"- Generated: {_format_ratio(answer_counts['generated'], answer_expected)}")
    if answer_counts["succeeded"]:
        print(f"- Succeeded: {_format_ratio(answer_counts['succeeded'], answer_expected)}")
    if answer_counts["ill_posed"]:
        print(f"- Ill-posed: {_format_ratio(answer_counts['ill_posed'], answer_expected)}")
    if answer_counts["failed"]:
        print(f"- Failed: {_format_ratio(answer_counts['failed'], answer_expected)}")

    print("\nCritiques:")
    total_expected = sum(critique_expected_by_mode.values())
    total_generated = sum(counts["generated"] for counts in critique_counts_by_mode.values())
    print(f"- Total generated: {_format_ratio(total_generated, total_expected)}")
    for mode in args.critique_modes:
        expected = critique_expected_by_mode.get(mode, 0)
        generated = critique_counts_by_mode.get(mode, Counter())["generated"]
        print(f"- {mode}: {_format_ratio(generated, expected)}")

    print("\nDebates:")
    print(f"- Ill-posed: {_format_ratio(actual_illposed['generated'], expected_illposed['expected'])}")
    total_expected_debates = expected_illposed["expected"] + sum(expected_critique_by_mode.values())
    total_actual_debates = actual_illposed["generated"] + sum(actual_critique_by_mode.values())
    print(f"- Critique total: {_format_ratio(sum(actual_critique_by_mode.values()), sum(expected_critique_by_mode.values()))}")
    for mode in args.critique_modes:
        expected = expected_critique_by_mode.get(mode, 0)
        actual = actual_critique_by_mode.get(mode, 0)
        print(f"- Critique {mode}: {_format_ratio(actual, expected)}")
    print(f"- Debates total: {_format_ratio(total_actual_debates, total_expected_debates)}")

    print("\nAutomated evaluations:")
    print(f"- Decisions: {_format_ratio(eval_actual, eval_expected)}")
    print(f"- Unique tasks: {len(judging_tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
