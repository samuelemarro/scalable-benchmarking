from collections import Counter, defaultdict
from pathlib import Path
import logging

from typing import Dict, List

from data_models import (
    AutomatedEvaluation,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_debate_entries,
    load_evaluation_entries,
    load_human_evaluation_entries,
)
from constants import (
    CLAIMANT_WIN_VERDICTS,
    CRITIQUE_VERDICT_CORRECT,
    DEFENDER_WIN_VERDICTS,
    STATUS_ILL_POSED,
)

logger = logging.getLogger(__name__)


def count_human_labels(evaluations_dir: Path):
    label_counts = defaultdict(int)
    for eval_file in evaluations_dir.glob("*.json"):
        data = load_human_evaluation_entries(eval_file)
        for dec in data.decisions:
            label_counts[dec.id] += 1
    return label_counts


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
                for idx, _ in enumerate(crit_ids):
                    claim_ids.add(f"critique/{mode_dir.name}/{q_slug}/{crit_file.stem}/{idx}")
    for debate_file in (debates_dir / "illposed").glob("*/*.json"):
        q_slug = debate_file.parent.name
        a_slug = debate_file.stem
        debates = load_debate_entries(debate_file)
        for idx, _ in enumerate(debates):
            claim_ids.add(f"illposed/{q_slug}/{a_slug}/{idx}")
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
                    verdict = attempts[-1].verdict
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


def collect_automated_evaluations(auto_eval_dir: Path) -> List[AutomatedEvaluation]:
    """Collect all automated evaluation decisions from all judge files."""
    all_decisions = []
    if not auto_eval_dir.exists():
        return all_decisions

    for eval_file in auto_eval_dir.glob("*.json"):
        data = load_evaluation_entries(eval_file)
        all_decisions.extend(data.decisions)

    return all_decisions


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
                for idx, entry in enumerate(entries):
                    attempts = entry.attempts if entry else None
                    verdict = attempts[-1].verdict if attempts else None
                    # Extract answer author and question author from entry
                    answer_author = entry.answer_author if entry else None
                    question_author = entry.question_author if entry else None
                    critic_model_name = entry.critic if entry else None

                    cid = f"critique/{mode_dir.name}/{q_slug}/{crit_file.stem}/{idx}"
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

    # For each model, track self-answer correctness percentages
    # Verdicts that mean "answer is correct": defender_wins_incorrect, defender_wins_minor
    # (Alice claimed error, but Bob defended successfully)
    model_self_answers = defaultdict(list)

    # Track self-answers declared "correct" by critics (no debate needed)
    model_self_answers_no_debate = defaultdict(int)

    # For each defender model (answer author), track their success rate when defending answers
    # Verdicts that side with defender: defender_wins_incorrect, defender_wins_minor
    model_as_defender_success_rate = defaultdict(list)

    # For each claimant model (critic), track their success rate when making claims
    # Verdicts that side with claimant: claimant_wins, mixed (partial win)
    model_as_claimant_success_rate = defaultdict(list)

    # For cross-model answers, track three categories per (answer_model, question_model) pair
    cross_model_answers = defaultdict(lambda: defaultdict(lambda: {
        "declared_correct": 0,  # Critic said "correct" (no debate)
        "critiqued_correct": 0,  # Critic said wrong, but majority of judges sided with defender
        "critiqued_wrong": 0,    # Critic said wrong, and majority of judges sided with critic
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
            # Count verdicts for this claim
            verdicts = [d.verdict for d in claim_decisions if d.verdict]

            if not verdicts:
                continue

            # Determine if this is a self-answer (question_model == answer_model)
            is_self_answer = (question_model == answer_model)

            if is_self_answer:
                # Calculate percentage of judges saying answer is correct
                # Correct = defender wins (defender_wins_incorrect or defender_wins_minor)
                correct_count = sum(1 for v in verdicts if v in DEFENDER_WIN_VERDICTS)
                percentage = 100 * correct_count / len(verdicts)
                model_self_answers[answer_model].append(percentage)
            else:
                # Cross-model answer - count judges siding with defender
                correct_count = sum(1 for v in verdicts if v in DEFENDER_WIN_VERDICTS)

                # Exclude ties from analysis (per user decision Q1.2)
                # Ties occur when correct_count == len(verdicts) / 2
                is_tie = (correct_count == len(verdicts) / 2)

                if not is_tie:
                    majority_correct = correct_count > len(verdicts) / 2
                    # This answer was critiqued (not declared "correct"), so categorize based on judge majority
                    if majority_correct:
                        cross_model_answers[answer_model][question_model]["critiqued_correct"] += 1
                    else:
                        cross_model_answers[answer_model][question_model]["critiqued_wrong"] += 1
                    cross_model_answers[answer_model][question_model]["total"] += 1
                # If tie, skip entirely (not counted in total)

            # Defender stats (Bob is the answer_model defending)
            defender_wins_count = sum(1 for v in verdicts if v in DEFENDER_WIN_VERDICTS)
            percentage = 100 * defender_wins_count / len(verdicts)
            model_as_defender_success_rate[answer_model].append(percentage)

            # Claimant stats (Alice is the critic_model claiming error)
            claimant_wins_count = sum(1 for v in verdicts if v in CLAIMANT_WIN_VERDICTS)
            percentage = 100 * claimant_wins_count / len(verdicts)
            model_as_claimant_success_rate[critic_model].append(percentage)

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


def print_agreement_stats(model_stats, title, total_judges=None):
    """Print average percentage of judges agreeing."""
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
        print(f"  {model}: {avg_percentage:.1f}% average (across {len(percentages)} critiques)")


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
                print(f"      {critiqued_correct_pct:.1f}% critiqued but judge majority says correct")
                print(f"      {critiqued_wrong_pct:.1f}% critiqued and judge majority says wrong")


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

    # Compute and print automated evaluation statistics
    model_self_answers, model_self_answers_no_debate, model_as_defender_success_rate, model_as_claimant_success_rate, cross_model_stats = compute_model_stats(auto_eval_dir, critiques_dir)

    print_agreement_stats(
        model_self_answers,
        "Self-answer correctness (debated - number of judges agreeing answer is correct)"
    )

    # Print self-answers declared correct without debate
    if model_self_answers_no_debate:
        print("\nSelf-answers declared correct by critics (no debate needed):")
        for model in sorted(model_self_answers_no_debate.keys()):
            count = model_self_answers_no_debate[model]
            print(f"  {model}: {count} answers")

    print_agreement_stats(
        model_as_defender_success_rate,
        "Defender success (number of judges siding with defender/answer)"
    )

    print_agreement_stats(
        model_as_claimant_success_rate,
        "Claimant success (number of judges siding with claimant/critic)"
    )

    print_cross_model_stats(cross_model_stats)


if __name__ == "__main__":
    main()
