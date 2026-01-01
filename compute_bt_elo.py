import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from constants import (
    CLAIMANT_WIN_VERDICTS,
    CRITIQUE_VERDICT_CORRECT,
    CRITIQUE_VERDICT_UNKNOWN,
    DEFENDER_WIN_VERDICTS,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
)
from data_models import (
    AutomatedEvaluation,
    BenchmarkEntry,
    load_answer_entries,
    load_benchmark_entries,
    load_critique_entries,
    load_evaluation_entries,
)
from model_config import load_registry


def final_question(entry: BenchmarkEntry) -> Optional[str]:
    generations = entry.generation_rounds or []
    if not generations:
        return None
    refinements = generations[-1].refinement_rounds or []
    if not refinements:
        return None
    return refinements[-1].question


def collect_decisions(auto_eval_dir: Path) -> Dict[str, List[AutomatedEvaluation]]:
    decisions_by_claim: Dict[str, List[AutomatedEvaluation]] = defaultdict(list)
    if not auto_eval_dir.exists():
        return decisions_by_claim
    for eval_file in auto_eval_dir.glob("*.json"):
        data = load_evaluation_entries(eval_file)
        for decision in data.decisions:
            if decision and decision.id:
                decisions_by_claim[decision.id].append(decision)
    return decisions_by_claim


def majority_side(decisions: Iterable[AutomatedEvaluation]) -> Optional[str]:
    claimant = defender = other = 0
    for decision in decisions:
        verdict = decision.verdict
        if verdict in CLAIMANT_WIN_VERDICTS:
            claimant += 1
        elif verdict in DEFENDER_WIN_VERDICTS:
            defender += 1
        else:
            other += 1
    total = claimant + defender + other
    if total == 0:
        return None
    if claimant > total / 2:
        return "claimant"
    if defender > total / 2:
        return "defender"
    return None


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
                    verdict = attempts[-1].verdict
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


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def log1pexp(x: float) -> float:
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def fit_bipartite_bt(
    pairs: List[Tuple[int, int, int, int]],
    num_answerers: int,
    num_questioners: int,
    max_iter: int,
    lr: float,
    tol: float,
    l2: float,
) -> Tuple[List[float], List[float]]:
    theta = [0.0 for _ in range(num_answerers)]
    psi = [0.0 for _ in range(num_questioners)]
    last_ll = None
    for step_idx in range(max_iter):
        grad_theta = [0.0 for _ in range(num_answerers)]
        grad_psi = [0.0 for _ in range(num_questioners)]
        ll = 0.0
        for a_idx, q_idx, wins, losses in pairs:
            total = wins + losses
            if total <= 0:
                continue
            diff = theta[a_idx] - psi[q_idx]
            p = sigmoid(diff)
            ll += wins * diff - total * log1pexp(diff)
            grad = wins - total * p
            grad_theta[a_idx] += grad
            grad_psi[q_idx] -= grad
        for i in range(num_answerers):
            grad_theta[i] -= l2 * theta[i]
        for j in range(num_questioners):
            grad_psi[j] -= l2 * psi[j]
        scale = lr / math.sqrt(step_idx + 1.0)
        max_step = 0.0
        for i in range(num_answerers):
            delta = scale * grad_theta[i]
            theta[i] += delta
            max_step = max(max_step, abs(delta))
        for j in range(num_questioners):
            delta = scale * grad_psi[j]
            psi[j] += delta
            max_step = max(max_step, abs(delta))
        if num_questioners:
            shift = sum(psi) / num_questioners
            if shift:
                psi = [v - shift for v in psi]
                theta = [v - shift for v in theta]
        if last_ll is not None and abs(ll - last_ll) < tol and max_step < tol:
            break
        last_ll = ll
    return theta, psi


def resolve_model(registry, name_or_slug: Optional[str]) -> Optional[str]:
    if not name_or_slug:
        return None
    if registry is None:
        return name_or_slug
    return registry.resolve_model_name(name_or_slug)


def collect_games(
    benchmarks_dir: Path,
    answers_dir: Path,
    critiques_dir: Path,
    auto_eval_dir: Path,
    registry_path: Optional[Path],
    answer_critique_mode: str,
    self_answer_critique_mode: str,
    fallback_any_mode: bool,
) -> Tuple[List[Tuple[str, str, int]], Counter]:
    registry = load_registry(str(registry_path)) if registry_path and registry_path.exists() else None
    critique_verdicts = load_critique_verdicts(critiques_dir)
    decisions_by_claim = collect_decisions(auto_eval_dir)
    skip_counts: Counter = Counter()
    games: List[Tuple[str, str, int]] = []

    for bench_path in benchmarks_dir.glob("*.json"):
        q_slug = bench_path.stem
        benchmarks = load_benchmark_entries(bench_path)
        q_name = resolve_model(registry, q_slug) or q_slug
        answers_root = answers_dir / q_slug
        if not answers_root.exists():
            continue
        for answer_file in answers_root.glob("*.json"):
            a_slug = answer_file.stem
            answers = load_answer_entries(answer_file)
            max_len = max(len(benchmarks), len(answers))
            for idx in range(max_len):
                bench_entry = benchmarks[idx] if idx < len(benchmarks) else None
                if not bench_entry or bench_entry.status != STATUS_SUCCEEDED:
                    skip_counts["question_missing_or_failed"] += 1
                    continue
                question_text = final_question(bench_entry)
                if not question_text:
                    skip_counts["question_missing_or_failed"] += 1
                    continue
                answer_entry = answers[idx] if idx < len(answers) else None
                if not answer_entry:
                    skip_counts["answer_missing"] += 1
                    continue

                answer_name = resolve_model(registry, answer_entry.answer_model) or resolve_model(registry, a_slug) or a_slug

                self_mode, self_verdict = find_critique_verdict(
                    critique_verdicts,
                    q_slug,
                    a_slug,
                    q_slug,
                    idx,
                    self_answer_critique_mode,
                    fallback_any_mode,
                )
                if self_verdict:
                    if self_verdict == CRITIQUE_VERDICT_CORRECT:
                        pass
                    elif self_verdict == CRITIQUE_VERDICT_UNKNOWN:
                        skip_counts["self_answer_unknown"] += 1
                        continue
                    else:
                        claim_id = f"critique/{self_mode}/{q_slug}/{a_slug}__{q_slug}/{idx}"
                        majority = majority_side(decisions_by_claim.get(claim_id, []))
                        if majority == "claimant":
                            skip_counts["self_answer_invalid"] += 1
                            continue
                        if majority is None:
                            skip_counts["self_answer_no_majority"] += 1
                            continue

                if answer_entry.status == STATUS_FAILED:
                    games.append((answer_name, q_name, 0))
                    continue

                if answer_entry.status == STATUS_ILL_POSED:
                    claim_id = f"illposed/{q_slug}/{a_slug}/{idx}"
                    majority = majority_side(decisions_by_claim.get(claim_id, []))
                    if majority == "claimant":
                        skip_counts["illposed_validated"] += 1
                        continue
                    if majority == "defender":
                        games.append((answer_name, q_name, 0))
                        continue
                    skip_counts["illposed_no_majority"] += 1
                    continue

                if answer_entry.status != STATUS_SUCCEEDED:
                    skip_counts["answer_invalid_status"] += 1
                    continue

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
                    skip_counts["critique_missing"] += 1
                    continue
                if verdict == CRITIQUE_VERDICT_UNKNOWN:
                    skip_counts["critique_unknown"] += 1
                    continue
                if verdict == CRITIQUE_VERDICT_CORRECT:
                    games.append((answer_name, q_name, 1))
                    continue

                claim_id = f"critique/{mode}/{q_slug}/{q_slug}__{a_slug}/{idx}"
                majority = majority_side(decisions_by_claim.get(claim_id, []))
                if majority == "defender":
                    games.append((answer_name, q_name, 1))
                    continue
                if majority == "claimant":
                    games.append((answer_name, q_name, 0))
                    continue
                skip_counts["critique_no_majority"] += 1

    return games, skip_counts


def build_pair_counts(games: List[Tuple[str, str, int]]) -> Dict[Tuple[str, str], List[int]]:
    counts: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
    for answerer, questioner, outcome in games:
        wins, losses = counts[(answerer, questioner)]
        if outcome == 1:
            wins += 1
        else:
            losses += 1
        counts[(answerer, questioner)] = [wins, losses]
    return counts


def render_ratings(
    role: str,
    names: List[str],
    logit_scores: Dict[str, float],
    wins: Dict[str, int],
    losses: Dict[str, int],
    base_elo: float,
    min_games: int,
) -> List[Tuple[str, float, float, int, int, int]]:
    scale = 400.0 / math.log(10.0)
    rows = []
    for name in names:
        w = wins.get(name, 0)
        l = losses.get(name, 0)
        total = w + l
        if total < min_games:
            continue
        logit = logit_scores.get(name, 0.0)
        elo = base_elo + scale * logit
        rows.append((name, elo, logit, w, l, total))
    rows.sort(key=lambda item: item[1], reverse=True)
    print(f"\n{role}:")
    for name, elo, _, w, l, total in rows:
        print(f"{elo:8.1f}  {w:4d}-{l:<4d}  ({total:4d})  {name}")
    return rows


def write_csv(path: Path, rows: List[Tuple[str, float, float, int, int, int]], role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["role", "model", "elo", "logit", "wins", "losses", "games"])
        for name, elo, logit, wins, losses, total in rows:
            writer.writerow([role, name, f"{elo:.2f}", f"{logit:.6f}", wins, losses, total])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute bipartite Bradley-Terry Elo ratings from benchmark results."
    )
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--automated-dir", type=Path, default=Path("automated_evaluations"))
    parser.add_argument("--config", type=Path, default=Path("configs/models.json"))
    parser.add_argument("--answer-critique-mode", type=str, default="evaluator")
    parser.add_argument("--self-answer-critique-mode", type=str, default="contradictor")
    parser.add_argument("--fallback-any-mode", action="store_true")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--base-elo", type=float, default=1000.0)
    parser.add_argument("--min-games", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    games, skips = collect_games(
        args.benchmarks_dir,
        args.answers_dir,
        args.critiques_dir,
        args.automated_dir,
        args.config,
        args.answer_critique_mode,
        args.self_answer_critique_mode,
        args.fallback_any_mode,
    )

    if not games:
        print("No valid games found after filtering.")
        if skips:
            print("Skipped:")
            for reason, count in sorted(skips.items()):
                print(f"  {reason}: {count}")
        return 1

    pair_counts = build_pair_counts(games)
    answerers = sorted({pair[0] for pair in pair_counts})
    questioners = sorted({pair[1] for pair in pair_counts})
    answerer_index = {name: idx for idx, name in enumerate(answerers)}
    questioner_index = {name: idx for idx, name in enumerate(questioners)}

    pairs = []
    for (answerer, questioner), (wins, losses) in pair_counts.items():
        pairs.append((answerer_index[answerer], questioner_index[questioner], wins, losses))

    theta, psi = fit_bipartite_bt(
        pairs,
        len(answerers),
        len(questioners),
        args.max_iter,
        args.lr,
        args.tol,
        args.l2,
    )

    answerer_scores = {name: theta[idx] for name, idx in answerer_index.items()}
    questioner_scores = {name: psi[idx] for name, idx in questioner_index.items()}

    answerer_wins = Counter()
    answerer_losses = Counter()
    questioner_wins = Counter()
    questioner_losses = Counter()
    for (answerer, questioner), (wins, losses) in pair_counts.items():
        answerer_wins[answerer] += wins
        answerer_losses[answerer] += losses
        questioner_wins[questioner] += losses
        questioner_losses[questioner] += wins

    total_games = sum(w + l for w, l in pair_counts.values())
    print(f"Games: {total_games}")
    if skips:
        print("Skipped:")
        for reason, count in sorted(skips.items()):
            print(f"  {reason}: {count}")

    answerer_rows = render_ratings(
        "Answerer Elo (Bob)",
        answerers,
        answerer_scores,
        answerer_wins,
        answerer_losses,
        args.base_elo,
        args.min_games,
    )
    questioner_rows = render_ratings(
        "Questioner Elo (Alice)",
        questioners,
        questioner_scores,
        questioner_wins,
        questioner_losses,
        args.base_elo,
        args.min_games,
    )

    if args.output_csv:
        write_csv(args.output_csv, answerer_rows, "answerer")
        write_csv(args.output_csv, questioner_rows, "questioner")
        print(f"\nWrote CSV to {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
