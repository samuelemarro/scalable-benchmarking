#!/usr/bin/env python3
"""
Check for issues in output files across the pipeline.

Reports files with failed statuses, unknown verdicts, parsing errors, etc.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Optional
from constants import (
    CRITIQUE_VERDICT_CORRECT,
    JUDGE_VERDICT_UNKNOWN,
    STATUS_FAILED,
    STATUS_ILL_POSED,
    STATUS_SUCCEEDED,
    VALID_CRITIQUE_VERDICTS,
    VALID_CRITIQUE_DEBATE_VERDICTS,
    VALID_ILLPOSED_DEBATE_VERDICTS,
    VALID_STATUSES,
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


def final_benchmark_question(entry: BenchmarkEntry) -> str:
    gen_rounds = entry.generation_rounds or []
    if not gen_rounds:
        return ""
    refinements = gen_rounds[-1].refinement_rounds or []
    if not refinements:
        return ""
    return refinements[-1].question or ""


def final_benchmark_answer(entry: BenchmarkEntry) -> str:
    gen_rounds = entry.generation_rounds or []
    if not gen_rounds:
        return ""
    refinements = gen_rounds[-1].refinement_rounds or []
    if not refinements:
        return ""
    return refinements[-1].answer or ""


def final_answer_text(entry: AnswerEntry) -> str:
    attempts = entry.attempts or []
    if not attempts:
        return ""
    return attempts[-1].answer or ""


def final_critique_text(entry: CritiqueEntry) -> str:
    attempts = entry.attempts or []
    if not attempts:
        return ""
    last = attempts[-1]
    return str(last.notes or last.raw_critique or "")


def _has_unknown_self_check(evaluation: Optional[Dict[str, Any]]) -> bool:
    if evaluation is None:
        return True
    if not isinstance(evaluation, dict):
        return True
    issues = evaluation.get("issues")
    return isinstance(issues, list) and "unknown" in issues


def check_benchmark_issues(file_path: Path) -> Dict[str, int]:
    """Check for issues in benchmark files."""
    issues = defaultdict(int)

    try:
        try:
            data = load_benchmark_entries(file_path)
        except json.JSONDecodeError:
            issues["parse_error"] += 1
            return issues
        except Exception as exc:
            if "Expected list" in str(exc):
                issues["invalid_format"] += 1
            else:
                issues["schema_error"] += 1
            return issues

        for entry in data:
            if not entry:
                continue
            status = entry.status
            if status == STATUS_FAILED:
                issues["failed_generation"] += 1
            elif status not in {STATUS_SUCCEEDED, STATUS_FAILED, STATUS_ILL_POSED}:
                issues["unknown_status"] += 1

            gen_rounds = entry.generation_rounds or []
            if gen_rounds:
                refinements = gen_rounds[-1].refinement_rounds or []
                if refinements and _has_unknown_self_check(refinements[-1].evaluation):
                    issues["unknown_self_check"] += 1

            # Check for empty questions
            final_q = final_benchmark_question(entry).strip()
            if not final_q and status == STATUS_SUCCEEDED:
                issues["empty_question"] += 1

            # Check for empty answers
            final_a = final_benchmark_answer(entry).strip()
            if not final_a and status == STATUS_SUCCEEDED:
                issues["empty_answer"] += 1
    except Exception:
        issues["parse_error"] += 1

    return issues


def check_answer_issues(file_path: Path, failed_answer_min_attempts: int = 5) -> Dict[str, int]:
    """Check for issues in answer files."""
    issues = defaultdict(int)

    try:
        try:
            data = load_answer_entries(file_path)
        except json.JSONDecodeError:
            issues["parse_error"] += 1
            return issues
        except Exception as exc:
            if "Expected list" in str(exc):
                issues["invalid_format"] += 1
            else:
                issues["schema_error"] += 1
            return issues

        for entry in data:
            if not entry:
                continue

            attempts = entry.attempts or []
            status = entry.status
            if status == STATUS_FAILED:
                if len(attempts) < failed_answer_min_attempts:
                    issues["failed_answer"] += 1
            elif status not in VALID_STATUSES:
                issues["unknown_status"] += 1

            if attempts and _has_unknown_self_check(attempts[-1].evaluation):
                issues["unknown_self_check"] += 1

            # Check for empty final answers (unless ill-posed)
            final_a = final_answer_text(entry).strip()
            if not final_a and status == STATUS_SUCCEEDED:
                issues["empty_answer"] += 1
    except Exception:
        issues["parse_error"] += 1

    return issues


def check_critique_issues(file_path: Path) -> Dict[str, int]:
    """Check for issues in critique files."""
    issues = defaultdict(int)

    try:
        try:
            data = load_critique_entries(file_path)
        except json.JSONDecodeError:
            issues["parse_error"] += 1
            return issues
        except Exception as exc:
            if "Expected list" in str(exc):
                issues["invalid_format"] += 1
            else:
                issues["schema_error"] += 1
            return issues

        for entry in data:
            if not entry:
                continue

            status = entry.status
            if status == STATUS_FAILED:
                issues["failed_critique"] += 1
            elif status not in {STATUS_SUCCEEDED, STATUS_FAILED}:
                issues["unknown_status"] += 1

            # Check verdict if present
            attempts = entry.attempts or []
            if attempts:
                verdict = attempts[-1].verdict
                if verdict and verdict not in VALID_CRITIQUE_VERDICTS:
                    issues["invalid_verdict"] += 1
                if _has_unknown_self_check(attempts[-1].evaluation):
                    issues["unknown_self_check"] += 1

            # Check for empty critiques (unless verdict is "correct")
            final_c = final_critique_text(entry).strip()
            verdict = attempts[-1].verdict if attempts else None
            if not final_c and status == STATUS_SUCCEEDED and verdict != CRITIQUE_VERDICT_CORRECT:
                issues["empty_critique"] += 1
    except Exception:
        issues["parse_error"] += 1

    return issues


def check_debate_issues(file_path: Path) -> Dict[str, int]:
    """Check for issues in debate files."""
    issues = defaultdict(int)

    try:
        try:
            data = load_debate_entries(file_path)
        except json.JSONDecodeError:
            issues["parse_error"] += 1
            return issues
        except Exception as exc:
            if "Expected list" in str(exc):
                issues["invalid_format"] += 1
            else:
                issues["schema_error"] += 1
            return issues

        for entry in data:
            if not entry:
                continue

            # Check for empty debates
            history = entry.history
            if not history:
                issues["empty_debate"] += 1

            # Check for malformed history entries
            for round_entry in history:
                if not round_entry.message or not round_entry.speaker:
                    issues["incomplete_round"] += 1
    except Exception:
        issues["parse_error"] += 1

    return issues


def check_evaluation_issues(file_path: Path) -> Dict[str, int]:
    """Check for issues in automated evaluation files."""
    issues = defaultdict(int)

    try:
        try:
            payload = load_evaluation_entries(file_path)
        except json.JSONDecodeError:
            issues["parse_error"] += 1
            return issues
        except Exception as exc:
            if "Invalid evaluation file format" in str(exc):
                issues["invalid_format"] += 1
            else:
                issues["schema_error"] += 1
            return issues

        decisions = payload.decisions

        for evaluation in decisions:
            verdict = evaluation.verdict

            if verdict == JUDGE_VERDICT_UNKNOWN:
                issues["unknown_verdict"] += 1
            elif verdict not in VALID_CRITIQUE_DEBATE_VERDICTS and verdict not in VALID_ILLPOSED_DEBATE_VERDICTS:
                issues["invalid_verdict"] += 1

            # Check for missing fields
            if not evaluation.reasoning:
                issues["missing_reasoning"] += 1
            if not evaluation.judge_model:
                issues["missing_judge"] += 1

            # Check confidence
            confidence = evaluation.confidence
            if confidence is not None:
                if not isinstance(confidence, int) or not (1 <= confidence <= 5):
                    issues["invalid_confidence"] += 1
                elif confidence <= 2:
                    issues["low_confidence"] += 1
            else:
                issues["missing_confidence"] += 1

            # Check status
            status = evaluation.status
            if status == STATUS_FAILED:
                issues["failed_evaluation"] += 1
            elif status not in {STATUS_SUCCEEDED, STATUS_FAILED, None}:
                issues["unknown_status"] += 1
    except Exception:
        issues["parse_error"] += 1

    return issues


def scan_directory(base_path: Path, check_func, pattern: str = "**/*.json") -> Dict[str, Dict[str, int]]:
    """Scan a directory for JSON files and check for issues."""
    results = {}

    if not base_path.exists():
        return results

    for file_path in base_path.glob(pattern):
        if file_path.is_file():
            issues = check_func(file_path)
            if issues:  # Only include files with issues
                rel_path = str(file_path.relative_to(base_path.parent))
                results[rel_path] = dict(issues)

    return results


def main():
    parser = argparse.ArgumentParser(description="Check output files for common issues.")
    parser.add_argument(
        "--failed-answer-min-attempts",
        type=int,
        default=5,
        help="Only flag failed answers with fewer than this many attempts.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("OUTPUT FILE ISSUE REPORT")
    print("=" * 80)
    print()

    total_files_with_issues = 0
    total_issues = 0

    # Check benchmarks
    print("BENCHMARKS")
    print("-" * 80)
    benchmark_issues = scan_directory(Path("benchmarks"), check_benchmark_issues)
    if benchmark_issues:
        for file_path, issues in sorted(benchmark_issues.items()):
            issue_count = sum(issues.values())
            total_files_with_issues += 1
            total_issues += issue_count
            print(f"\n{file_path}:")
            for issue_type, count in sorted(issues.items()):
                print(f"  - {issue_type}: {count}")
    else:
        print("No issues found.")
    print()

    # Check answers
    print("ANSWERS")
    print("-" * 80)
    answer_issues = scan_directory(
        Path("answers"),
        lambda path: check_answer_issues(path, args.failed_answer_min_attempts),
    )
    if answer_issues:
        for file_path, issues in sorted(answer_issues.items()):
            issue_count = sum(issues.values())
            total_files_with_issues += 1
            total_issues += issue_count
            print(f"\n{file_path}:")
            for issue_type, count in sorted(issues.items()):
                print(f"  - {issue_type}: {count}")
    else:
        print("No issues found.")
    print()

    # Check critiques
    print("CRITIQUES")
    print("-" * 80)
    critique_issues = scan_directory(Path("critiques"), check_critique_issues)
    if critique_issues:
        for file_path, issues in sorted(critique_issues.items()):
            issue_count = sum(issues.values())
            total_files_with_issues += 1
            total_issues += issue_count
            print(f"\n{file_path}:")
            for issue_type, count in sorted(issues.items()):
                print(f"  - {issue_type}: {count}")
    else:
        print("No issues found.")
    print()

    # Check debates
    print("DEBATES")
    print("-" * 80)
    debate_issues = scan_directory(Path("debates"), check_debate_issues)
    if debate_issues:
        for file_path, issues in sorted(debate_issues.items()):
            issue_count = sum(issues.values())
            total_files_with_issues += 1
            total_issues += issue_count
            print(f"\n{file_path}:")
            for issue_type, count in sorted(issues.items()):
                print(f"  - {issue_type}: {count}")
    else:
        print("No issues found.")
    print()

    # Check automated evaluations
    print("AUTOMATED EVALUATIONS")
    print("-" * 80)
    eval_issues = scan_directory(Path("automated_evaluations"), check_evaluation_issues)
    if eval_issues:
        for file_path, issues in sorted(eval_issues.items()):
            issue_count = sum(issues.values())
            total_files_with_issues += 1
            total_issues += issue_count
            print(f"\n{file_path}:")
            for issue_type, count in sorted(issues.items()):
                print(f"  - {issue_type}: {count}")
    else:
        print("No issues found.")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files with issues: {total_files_with_issues}")
    print(f"Total issues found: {total_issues}")
    print()

    # Issue type breakdown
    all_issue_types = defaultdict(int)
    for issues_dict in [benchmark_issues, answer_issues, critique_issues, debate_issues, eval_issues]:
        for file_issues in issues_dict.values():
            for issue_type, count in file_issues.items():
                all_issue_types[issue_type] += count

    if all_issue_types:
        print("Issue breakdown by type:")
        for issue_type, count in sorted(all_issue_types.items(), key=lambda x: -x[1]):
            print(f"  - {issue_type}: {count}")

    print()


if __name__ == "__main__":
    main()
