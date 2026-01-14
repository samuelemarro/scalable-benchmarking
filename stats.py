from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import logging

from typing import Dict, List, Optional, Tuple

from data_models import (
    AutomatedEvaluation,
    BenchmarkEntry,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_evaluation_entries,
    load_human_evaluation_entries,
)
from constants import (
    CRITIQUE_VERDICT_CORRECT,
    CRITIQUE_VERDICT_INCORRECT,
    CRITIQUE_VERDICT_INSUFFICIENT,
    CRITIQUE_VERDICT_OBSCURE,
    CRITIQUE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
)
from victory import (
    VictorySide,
    resolve_automated_victory,
)

logger = logging.getLogger(__name__)


def count_human_labels(evaluations_dir: Path):
    label_counts = defaultdict(int)
    for eval_file in evaluations_dir.glob("*.json"):
        data = load_human_evaluation_entries(eval_file)
        for dec in data.decisions:
            label_counts[dec.id] += 1
    return label_counts


def _task_id(prefix: str, run_id: Optional[str], topic_slug: Optional[str], question: Optional[str]) -> str:
    if run_id:
        return f"{prefix}/{run_id}"
    if topic_slug and question:
        digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}/{topic_slug}/{digest}"
    return f"{prefix}/unknown"


_CRITIQUE_INCORRECT_EQUIV = {
    CRITIQUE_VERDICT_INCORRECT,
    CRITIQUE_VERDICT_INSUFFICIENT,
    CRITIQUE_VERDICT_OBSCURE,
}


def normalize_critique_verdict(verdict: Optional[str]) -> Optional[str]:
    if verdict in _CRITIQUE_INCORRECT_EQUIV:
        return CRITIQUE_VERDICT_INCORRECT
    return verdict


def final_question(entry: BenchmarkEntry) -> Optional[str]:
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question


def load_critique_verdicts(critiques_dir: Path) -> Dict[Tuple[str, str, str, int], Dict[str, str]]:
    verdicts: Dict[Tuple[str, str, str, int], Dict[str, str]] = defaultdict(dict)
    if not critiques_dir.exists():
        return verdicts
    for mode_dir in critiques_dir.glob("*"):
        mode = mode_dir.name
        for q_dir in mode_dir.glob("*"):
            q_slug = q_dir.name
            for crit_file in q_dir.glob("*.json"):
                parts = crit_file.stem.split("__", 1)
                if len(parts) != 2:
                    continue
                critic_slug, answer_slug = parts
                entries = load_critique_entries(crit_file)
                for idx, entry in enumerate(entries):
                    if not entry or entry.status != STATUS_SUCCEEDED:
                        continue
                    attempts = entry.attempts or []
                    if not attempts:
                        continue
                    verdict = normalize_critique_verdict(attempts[-1].verdict)
                    if not verdict:
                        continue
                    key = (q_slug, critic_slug, answer_slug, idx)
                    verdicts[key][mode] = verdict
    return verdicts


def find_critique_verdict(
    verdicts: Dict[Tuple[str, str, str, int], Dict[str, str]],
    q_slug: str,
    critic_slug: str,
    answer_slug: str,
    idx: int,
    preferred_mode: Optional[str],
    fallback_any: bool,
) -> Tuple[Optional[str], Optional[str]]:
    key = (q_slug, critic_slug, answer_slug, idx)
    modes = verdicts.get(key, {})
    if preferred_mode and preferred_mode in modes:
        return preferred_mode, modes[preferred_mode]
    if not fallback_any:
        return None, None
    if not modes:
        return None, None
    mode = sorted(modes.keys())[0]
    return mode, modes[mode]


def count_items(path: Path, kind: str):
    counts = 0
    if kind == "questions":
        for file in path.glob("*.json"):
            entries = load_benchmark_entries(file)
            counts += len(entries)
    elif kind == "answers":
        for q_dir in path.glob("*"):
            for ans_file in q_dir.glob("*.json"):
                entries = load_answer_entries(ans_file)
                counts += len(entries)
    elif kind in {"critiques", "illposed"}:
        base = "critiques" if kind == "critiques" else "debates/illposed"
        base_path = Path(base)
        for sub in base_path.glob("**/*.json"):
            entries = load_debate_entries(sub) if kind == "illposed" else load_critique_entries(sub)
            counts += len(entries)
    return counts


def collect_claim_ids(critiques_dir: Path, debates_dir: Path):
    claim_ids = set()
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                q_slug = q_dir.name
                crit_ids = load_critique_entries(crit_file)
                for entry in crit_ids:
                    if not entry:
                        continue
                    prefix = f"critique/{mode_dir.name}/{q_slug}/{crit_file.stem}"
                    claim_ids.add(_task_id(prefix, entry.run_id, entry.topic_slug, entry.question))
    for debate_file in (debates_dir / "illposed").glob("*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        debates = load_debate_entries(debate_file)
        for entry in debates:
            if not entry:
                continue
            prefix = f"illposed/{q_slug}/{a_slug}"
            claim_ids.add(_task_id(prefix, entry.run_id, entry.topic_slug, entry.question))
    return claim_ids


def critique_verdicts(critiques_dir: Path):
    verdict_counts = Counter()
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                entries = load_critique_entries(crit_file)
                for entry in entries:
                    if not entry:
                        verdict_counts["missing"] += 1
                        continue
                    attempts = entry.attempts or []
                    if not attempts:
                        verdict_counts["missing"] += 1
                        continue
                    verdict = normalize_critique_verdict(attempts[-1].verdict)
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
            entries = load_answer_entries(ans_file)
            for entry in entries:
                if entry and entry.status == STATUS_ILL_POSED:
                    count += 1
    return count


def collect_decisions_by_claim(auto_eval_dir: Path) -> Dict[str, List[AutomatedEvaluation]]:
    decisions_by_claim: Dict[str, List[AutomatedEvaluation]] = defaultdict(list)
    for decision in collect_automated_evaluations(auto_eval_dir):
        if decision.id:
            decisions_by_claim[decision.id].append(decision)
    return decisions_by_claim


def _format_count(count: int, total: int) -> str:
    if total <= 0:
        return f"{count} (0.00%)"
    pct = 100.0 * count / total
    return f"{count} ({pct:.2f}%)"


def print_protocol_stats(
    benchmarks_dir: Path,
    answers_dir: Path,
    critiques_dir: Path,
    auto_eval_dir: Path,
    answer_critique_mode: str = "evaluator",
    self_answer_critique_mode: str = "contradictor",
    fallback_any_mode: bool = False,
) -> None:
    critique_verdicts = load_critique_verdicts(critiques_dir)
    decisions_by_claim = collect_decisions_by_claim(auto_eval_dir)

    counts = Counter()
    candidate_games = 0

    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        benchmarks = load_benchmark_entries(bench_path)
        answers_root = answers_dir / q_slug
        if not answers_root.exists():
            continue
        for answer_file in answers_root.glob("*.json"):
            a_slug = answer_file.stem
            answers = load_answer_entries(answer_file)
            max_len = max(len(benchmarks), len(answers))
            for idx in range(max_len):
                counts["pairs_total"] += 1
                bench_entry = benchmarks[idx] if idx < len(benchmarks) else None
                if not bench_entry or bench_entry.status != STATUS_SUCCEEDED:
                    counts["drop_question_missing_or_failed"] += 1
                    continue
                question_text = final_question(bench_entry)
                if not question_text:
                    counts["drop_question_missing_or_failed"] += 1
                    continue
                answer_entry = answers[idx] if idx < len(answers) else None
                if not answer_entry:
                    counts["drop_answer_missing"] += 1
                    continue

                candidate_games += 1

                # Step 3: Bob critiques Alice's self-answer (question validity check)
                self_mode, self_verdict = find_critique_verdict(
                    critique_verdicts,
                    q_slug,
                    a_slug,
                    q_slug,
                    idx,
                    self_answer_critique_mode,
                    fallback_any_mode,
                )
                if not self_verdict:
                    counts["drop_self_critique_missing"] += 1
                    counts["final_drop"] += 1
                    continue
                counts["self_critique_present"] += 1
                self_verdict = normalize_critique_verdict(self_verdict)

                if self_verdict == CRITIQUE_VERDICT_UNKNOWN:
                    counts["drop_self_critique_unknown"] += 1
                    counts["final_drop"] += 1
                    continue

                if self_verdict == CRITIQUE_VERDICT_CORRECT:
                    counts["self_critique_correct"] += 1
                else:
                    claim_id = f"critique/{self_mode}/{q_slug}/{a_slug}__{q_slug}/{idx}"
                    outcome = resolve_automated_victory(
                        "critique",
                        decisions_by_claim.get(claim_id, []),
                        context=claim_id,
                    )
                    if outcome == VictorySide.ALICE:
                        counts["drop_self_critique_upheld"] += 1
                        counts["final_drop"] += 1
                        continue
                    if outcome == VictorySide.BOB:
                        counts["self_critique_rejected"] += 1
                    else:
                        counts["drop_self_critique_no_victory"] += 1
                        counts["final_drop"] += 1
                        continue

                # Step 4: Bob answers Alice's question
                if answer_entry.status == STATUS_FAILED:
                    counts["answer_failed"] += 1
                    counts["final_alice_wins"] += 1
                    continue

                if answer_entry.status == STATUS_ILL_POSED:
                    counts["answer_illposed_claimed"] += 1
                    claim_id = f"illposed/{q_slug}/{a_slug}/{idx}"
                    outcome = resolve_automated_victory(
                        "illposed",
                        decisions_by_claim.get(claim_id, []),
                        context=claim_id,
                    )
                    if outcome == VictorySide.ALICE:
                        counts["drop_illposed_upheld"] += 1
                        counts["final_drop"] += 1
                        continue
                    if outcome == VictorySide.BOB:
                        counts["illposed_rejected"] += 1
                        counts["final_alice_wins"] += 1
                        continue
                    counts["drop_illposed_no_victory"] += 1
                    counts["final_drop"] += 1
                    continue

                if answer_entry.status != STATUS_SUCCEEDED:
                    counts["drop_answer_invalid_status"] += 1
                    counts["final_drop"] += 1
                    continue

                counts["answer_succeeded"] += 1

                # Step 5: Alice critiques Bob's answer
                mode, verdict = find_critique_verdict(
                    critique_verdicts,
                    q_slug,
                    q_slug,
                    a_slug,
                    idx,
                    answer_critique_mode,
                    fallback_any_mode,
                )
                if not verdict:
                    counts["drop_critique_missing"] += 1
                    counts["final_drop"] += 1
                    continue
                verdict = normalize_critique_verdict(verdict)
                if verdict == CRITIQUE_VERDICT_UNKNOWN:
                    counts["drop_critique_unknown"] += 1
                    counts["final_drop"] += 1
                    continue
                if verdict == CRITIQUE_VERDICT_CORRECT:
                    counts["critique_says_correct"] += 1
                    counts["final_bob_wins"] += 1
                    continue

                claim_id = f"critique/{mode}/{q_slug}/{q_slug}__{a_slug}/{idx}"
                outcome = resolve_automated_victory(
                    "critique",
                    decisions_by_claim.get(claim_id, []),
                    context=claim_id,
                )
                if outcome == VictorySide.BOB:
                    counts["critique_incorrect_defender_wins"] += 1
                    counts["final_bob_wins"] += 1
                    continue
                if outcome == VictorySide.ALICE:
                    counts["critique_incorrect_claimant_wins"] += 1
                    counts["final_alice_wins"] += 1
                    continue
                counts["critique_incorrect_no_victory"] += 1
                counts["final_drop"] += 1

    pairs_total = counts["pairs_total"]

    print("\nProtocol flow (percentages of candidate games):")
    print(f"  Total pairs considered: {pairs_total}")
    print(f"  Candidate games (valid question + answer entry): {_format_count(candidate_games, pairs_total)}")
    print(f"  Dropped before candidate (question missing/failed): {_format_count(counts['drop_question_missing_or_failed'], pairs_total)}")
    print(f"  Dropped before candidate (answer missing): {_format_count(counts['drop_answer_missing'], pairs_total)}")

    print(f"  Bob critiques Alice's question: {_format_count(counts['self_critique_present'], candidate_games)}")
    print(f"  Bob's critique missing: {_format_count(counts['drop_self_critique_missing'], candidate_games)}")
    print(f"  Bob's critique says correct: {_format_count(counts['self_critique_correct'], candidate_games)}")
    print(f"  Bob's critique unknown: {_format_count(counts['drop_self_critique_unknown'], candidate_games)}")
    print(f"  Bob's critique upheld (drop): {_format_count(counts['drop_self_critique_upheld'], candidate_games)}")
    print(f"  Bob's critique rejected: {_format_count(counts['self_critique_rejected'], candidate_games)}")
    print(f"  Bob's critique no victory (drop): {_format_count(counts['drop_self_critique_no_victory'], candidate_games)}")

    print(f"  Bob fails to answer: {_format_count(counts['answer_failed'], candidate_games)}")
    print(f"  Bob claims ill-posed: {_format_count(counts['answer_illposed_claimed'], candidate_games)}")
    print(f"  Ill-posed upheld (drop): {_format_count(counts['drop_illposed_upheld'], candidate_games)}")
    print(f"  Ill-posed rejected (Alice wins): {_format_count(counts['illposed_rejected'], candidate_games)}")
    print(f"  Ill-posed no victory (drop): {_format_count(counts['drop_illposed_no_victory'], candidate_games)}")
    print(f"  Bob answers successfully: {_format_count(counts['answer_succeeded'], candidate_games)}")

    print(f"  Alice's critique missing: {_format_count(counts['drop_critique_missing'], candidate_games)}")
    print(f"  Alice's critique unknown: {_format_count(counts['drop_critique_unknown'], candidate_games)}")
    print(f"  Alice says answer correct (Bob wins): {_format_count(counts['critique_says_correct'], candidate_games)}")
    print(f"  Alice says incorrect, Bob wins: {_format_count(counts['critique_incorrect_defender_wins'], candidate_games)}")
    print(f"  Alice says incorrect, Alice wins: {_format_count(counts['critique_incorrect_claimant_wins'], candidate_games)}")
    print(f"  Alice says incorrect, no victory (drop): {_format_count(counts['critique_incorrect_no_victory'], candidate_games)}")

    print(f"  Final Bob wins: {_format_count(counts['final_bob_wins'], candidate_games)}")
    print(f"  Final Alice wins: {_format_count(counts['final_alice_wins'], candidate_games)}")
    print(f"  Final dropped: {_format_count(counts['final_drop'], candidate_games)}")


def collect_automated_evaluations(auto_eval_dir: Path) -> List[AutomatedEvaluation]:
    """Collect all automated evaluation decisions from all judge files."""
    all_decisions = []
    if not auto_eval_dir.exists():
        return all_decisions

    for eval_file in auto_eval_dir.glob("*.json"):
        data = load_evaluation_entries(eval_file)
        all_decisions.extend(data.decisions)

    return all_decisions


def _critique_target_key(claim_id: str) -> Optional[str]:
    parts = claim_id.split("/")
    if len(parts) < 5:
        return None
    q_slug = parts[-3]
    critic_and_answer = parts[-2]
    token = parts[-1]
    if "__" not in critic_and_answer:
        return None
    _, answer_slug = critic_and_answer.split("__", 1)
    return f"critique/{q_slug}/{answer_slug}/{token}"


def _illposed_target_key(claim_id: str) -> Optional[str]:
    parts = claim_id.split("/")
    if len(parts) < 4:
        return None
    q_slug = parts[-3]
    token = parts[-1]
    return f"illposed/{q_slug}/{token}"


def count_inter_judge_disagreements(auto_eval_dir: Path) -> Tuple[int, int]:
    """Count claims with non-unanimous judge verdicts."""
    decisions = collect_automated_evaluations(auto_eval_dir)
    decisions_by_claim = defaultdict(list)
    for decision in decisions:
        if decision.id:
            decisions_by_claim[decision.id].append(decision)

    naive_count = 0
    dedup_targets = set()

    for claim_id, claim_decisions in decisions_by_claim.items():
        if not claim_decisions:
            continue
        claim_type = claim_decisions[0].type
        if claim_type not in {"critique", "critique_debate", "illposed"}:
            continue
        verdicts = [d.verdict for d in claim_decisions if d.verdict]
        if len(verdicts) < 2:
            continue
        if len(set(verdicts)) <= 1:
            continue

        naive_count += 1
        if claim_type in {"critique", "critique_debate"}:
            target_key = _critique_target_key(claim_id)
        else:
            target_key = _illposed_target_key(claim_id)
        dedup_targets.add(target_key or f"{claim_type}/{claim_id}")

    return naive_count, len(dedup_targets)


def build_critique_verdict_map(critiques_dir: Path) -> Dict[str, Dict]:
    """
    Build a map of critique IDs to verdict metadata.
    This is used by both compute_model_stats and main to avoid duplication.

    Returns:
        Dict mapping critique IDs to dicts containing:
        - verdict: str (correct/incorrect/insufficient/obscure)
        - answer_author: str (model name)
        - question_author: str (model name)
        - critic: str (model name)
    """
    critique_verdict_map = {}
    for mode_dir in critiques_dir.glob("*"):
        for q_dir in mode_dir.glob("*"):
            for crit_file in q_dir.glob("*.json"):
                q_slug = q_dir.name
                entries = load_critique_entries(crit_file)
                for entry in entries:
                    if not entry:
                        continue
                    attempts = entry.attempts if entry else None
                    verdict = normalize_critique_verdict(attempts[-1].verdict) if attempts else None
                    # Extract answer author and question author from entry
                    answer_author = entry.answer_author if entry else None
                    question_author = entry.question_author if entry else None
                    critic_model_name = entry.critic if entry else None

                    prefix = f"critique/{mode_dir.name}/{q_slug}/{crit_file.stem}"
                    cid = _task_id(prefix, entry.run_id, entry.topic_slug, entry.question)
                    critique_verdict_map[cid] = {
                        "verdict": verdict,
                        "answer_author": answer_author,
                        "question_author": question_author,
                        "critic": critic_model_name
                    }
    return critique_verdict_map


def compute_model_stats(auto_eval_dir: Path, critiques_dir: Path):
    """Compute agreement statistics for various model roles."""
    decisions = collect_automated_evaluations(auto_eval_dir)

    # Track all encountered model names for validation
    encountered_models = set()

    # Group decisions by claim ID
    decisions_by_claim = defaultdict(list)
    for decision in decisions:
        claim_id = decision.id
        if claim_id:
            decisions_by_claim[claim_id].append(decision)
            # Track model names
            for key in ["question_model", "answer_model", "critic_model", "judge_model"]:
                model_name = getattr(decision, key)
                if model_name:
                    encountered_models.add(model_name)

    # For each model, track self-answer correctness victory rates.
    model_self_answers = defaultdict(list)

    # Track self-answers declared "correct" by critics (no debate needed)
    model_self_answers_no_debate = defaultdict(int)

    # For each defender model (answer author), track their victory rate when defending answers.
    model_as_defender_success_rate = defaultdict(list)

    # For each claimant model (critic), track their victory rate when making claims.
    model_as_claimant_success_rate = defaultdict(list)

    # For cross-model answers, track three categories per (answer_model, question_model) pair
    cross_model_answers = defaultdict(lambda: defaultdict(lambda: {
        "declared_correct": 0,  # Critic said "correct" (no debate)
        "critiqued_correct": 0,  # Critic said wrong, but victory rules sided with defender
        "critiqued_wrong": 0,    # Critic said wrong, and victory rules sided with claimant
        "total": 0
    }))

    # Build critique verdict map to identify "correct" verdicts (no debate)
    critique_verdict_map = build_critique_verdict_map(critiques_dir)

    for claim_id, claim_decisions in decisions_by_claim.items():
        if not claim_decisions:
            continue

        # Extract metadata from first decision (all should have same metadata)
        first = claim_decisions[0]
        claim_type = first.type
        question_model = first.question_model
        answer_model = first.answer_model
        critic_model = first.critic_model
        mode = first.mode

        if claim_type == "critique":
            outcome = resolve_automated_victory(claim_type, claim_decisions, context=claim_id)
            if outcome in {None, VictorySide.DROP}:
                continue

            # Determine if this is a self-answer (question_model == answer_model)
            is_self_answer = (question_model == answer_model)

            if is_self_answer:
                win_rate = 100.0 if outcome == VictorySide.BOB else 0.0
                model_self_answers[answer_model].append(win_rate)
            else:
                if outcome == VictorySide.BOB:
                    cross_model_answers[answer_model][question_model]["critiqued_correct"] += 1
                else:
                    cross_model_answers[answer_model][question_model]["critiqued_wrong"] += 1
                cross_model_answers[answer_model][question_model]["total"] += 1

            # Defender stats (Bob is the answer_model defending)
            defender_rate = 100.0 if outcome == VictorySide.BOB else 0.0
            model_as_defender_success_rate[answer_model].append(defender_rate)

            # Claimant stats (Alice is the critic_model claiming error)
            claimant_rate = 100.0 if outcome == VictorySide.ALICE else 0.0
            model_as_claimant_success_rate[critic_model].append(claimant_rate)

    # Now add answers declared "correct" by critics (these don't appear in automated evaluations)
    for cid, crit_info in critique_verdict_map.items():
        if crit_info["verdict"] == CRITIQUE_VERDICT_CORRECT:
            answer_author = crit_info["answer_author"]
            question_author = crit_info["question_author"]

            if answer_author and question_author:
                if answer_author == question_author:
                    # Self-answer declared correct (no debate needed)
                    model_self_answers_no_debate[answer_author] += 1
                else:
                    # Cross-model answer declared correct
                    cross_model_answers[answer_author][question_author]["declared_correct"] += 1
                    cross_model_answers[answer_author][question_author]["total"] += 1

    # Log all model names encountered for verification
    if encountered_models:
        logger.info(f"Encountered {len(encountered_models)} unique model names in data: {sorted(encountered_models)}")

    return model_self_answers, model_self_answers_no_debate, model_as_defender_success_rate, model_as_claimant_success_rate, cross_model_answers


def print_agreement_stats(model_stats, title):
    """Print average victory rate across claims."""
    if not model_stats:
        print(f"\n{title}: No data available")
        return

    print(f"\n{title}:")
    for model in sorted(model_stats.keys()):
        percentages = model_stats[model]
        if not percentages:
            continue

        # Safe division with explicit check
        avg_percentage = sum(percentages) / len(percentages) if percentages else 0.0
        print(f"  {model}: {avg_percentage:.1f}% victory rate (across {len(percentages)} critiques)")


def print_cross_model_stats(cross_model_stats):
    """Print statistics for models answering other models' questions."""
    if not cross_model_stats:
        print("\nCross-model answer correctness: No data available")
        return

    print("\nCross-model answer correctness (by question maker):")
    for answer_model in sorted(cross_model_stats.keys()):
        print(f"\n  {answer_model} answering:")
        by_q_model = cross_model_stats[answer_model]
        for q_model in sorted(by_q_model.keys()):
            stats = by_q_model[q_model]
            total = stats["total"]
            if total > 0:
                declared_pct = 100 * stats["declared_correct"] / total
                critiqued_correct_pct = 100 * stats["critiqued_correct"] / total
                critiqued_wrong_pct = 100 * stats["critiqued_wrong"] / total

                print(f"    {q_model}'s questions ({total} total):")
                print(f"      {declared_pct:.1f}% declared correct by critic (no debate)")
                print(f"      {critiqued_correct_pct:.1f}% critiqued but victory rules say correct")
                print(f"      {critiqued_wrong_pct:.1f}% critiqued and victory rules say wrong")


def main():
    benchmarks_dir = Path("benchmarks")
    answers_dir = Path("answers")
    critiques_dir = Path("critiques")
    debates_dir = Path("debates")
    evaluations_dir = Path("evaluations")
    auto_eval_dir = Path("automated_evaluations")

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

    # Build shared verdict map (reused by compute_model_stats)
    critique_verdict_map = build_critique_verdict_map(critiques_dir)

    for cid in claim_ids:
        if cid.startswith("critique/"):
            crit_info = critique_verdict_map.get(cid, {})
            v = crit_info.get("verdict") if isinstance(crit_info, dict) else crit_info
            if v == CRITIQUE_VERDICT_CORRECT:
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

    inter_judge_naive, inter_judge_dedup = count_inter_judge_disagreements(auto_eval_dir)
    print("\nInter-judge disagreements (critiques/ill-posed claims):")
    print(f"  naive: {inter_judge_naive}")
    print(f"  deduped by answer/question: {inter_judge_dedup}")

    print_protocol_stats(benchmarks_dir, answers_dir, critiques_dir, auto_eval_dir)

    # Compute and print automated evaluation statistics
    model_self_answers, model_self_answers_no_debate, model_as_defender_success_rate, model_as_claimant_success_rate, cross_model_stats = compute_model_stats(auto_eval_dir, critiques_dir)

    print_agreement_stats(
        model_self_answers,
        "Self-answer correctness (debated victory rate for answer correctness)"
    )

    # Print self-answers declared correct without debate
    if model_self_answers_no_debate:
        print("\nSelf-answers declared correct by critics (no debate needed):")
        for model in sorted(model_self_answers_no_debate.keys()):
            count = model_self_answers_no_debate[model]
            print(f"  {model}: {count} answers")

    print_agreement_stats(
        model_as_defender_success_rate,
        "Defender success (victory rate for defender/answer)"
    )

    print_agreement_stats(
        model_as_claimant_success_rate,
        "Claimant success (victory rate for claimant/critic)"
    )

    print_cross_model_stats(cross_model_stats)


if __name__ == "__main__":
    main()
