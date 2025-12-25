#!/usr/bin/env python3
"""
Check for issues in output files across the pipeline.

Reports files with failed statuses, unknown verdicts, parsing errors, etc.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from utils import load_json


def check_benchmark_issues(file_path: Path) -> Dict[str, int]:
    """Check for issues in benchmark files."""
    issues = defaultdict(int)

    try:
        data = load_json(file_path, [])
        if not isinstance(data, list):
            issues["invalid_format"] += 1
            return issues

        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                issues["invalid_entry"] += 1
                continue

            status = entry.get("status")
            if status == "failed":
                issues["failed_generation"] += 1
            elif status not in ["succeeded", "failed"]:
                issues["unknown_status"] += 1

            # Check for empty questions
            final_q = entry.get("final_question", "").strip()
            if not final_q and status == "succeeded":
                issues["empty_question"] += 1

            # Check for empty answers
            final_a = entry.get("final_answer", "").strip()
            if not final_a and status == "succeeded":
                issues["empty_answer"] += 1

    except Exception as e:
        issues["parse_error"] += 1

    return issues


def check_answer_issues(file_path: Path) -> Dict[str, int]:
    """Check for issues in answer files."""
    issues = defaultdict(int)

    try:
        data = load_json(file_path, [])
        if not isinstance(data, list):
            issues["invalid_format"] += 1
            return issues

        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                issues["invalid_entry"] += 1
                continue

            status = entry.get("status")
            if status == "failed":
                issues["failed_answer"] += 1
            elif status not in ["succeeded", "failed"]:
                issues["unknown_status"] += 1

            # Check for empty final answers
            final_a = entry.get("final_answer", "").strip()
            if not final_a and status == "succeeded":
                issues["empty_answer"] += 1

    except Exception as e:
        issues["parse_error"] += 1

    return issues


def check_critique_issues(file_path: Path) -> Dict[str, int]:
    """Check for issues in critique files."""
    issues = defaultdict(int)

    try:
        data = load_json(file_path, [])
        if not isinstance(data, list):
            issues["invalid_format"] += 1
            return issues

        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                issues["invalid_entry"] += 1
                continue

            status = entry.get("status")
            if status == "failed":
                issues["failed_critique"] += 1
            elif status not in ["succeeded", "failed"]:
                issues["unknown_status"] += 1

            # Check for empty critiques
            final_c = entry.get("final_critique", "").strip()
            if not final_c and status == "succeeded":
                issues["empty_critique"] += 1

    except Exception as e:
        issues["parse_error"] += 1

    return issues


def check_debate_issues(file_path: Path) -> Dict[str, int]:
    """Check for issues in debate files."""
    issues = defaultdict(int)

    try:
        data = load_json(file_path, [])
        if not isinstance(data, list):
            issues["invalid_format"] += 1
            return issues

        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                issues["invalid_entry"] += 1
                continue

            # Check for empty debates
            history = entry.get("history", [])
            if not history:
                issues["empty_debate"] += 1

            # Check for malformed history entries
            for round_entry in history:
                if not isinstance(round_entry, dict):
                    issues["invalid_round"] += 1
                    continue
                if "message" not in round_entry or "speaker" not in round_entry:
                    issues["incomplete_round"] += 1

    except Exception as e:
        issues["parse_error"] += 1

    return issues


def check_evaluation_issues(file_path: Path) -> Dict[str, int]:
    """Check for issues in automated evaluation files."""
    issues = defaultdict(int)

    try:
        data = load_json(file_path, {})
        if not isinstance(data, dict):
            issues["invalid_format"] += 1
            return issues

        for task_id, evaluation in data.items():
            if not isinstance(evaluation, dict):
                issues["invalid_entry"] += 1
                continue

            verdict = evaluation.get("verdict")
            if verdict == "unknown":
                issues["unknown_verdict"] += 1
            elif verdict not in ["defender_wins", "claimant_wins", "tie", "unknown"]:
                issues["invalid_verdict"] += 1

            # Check for missing fields
            if not evaluation.get("reasoning"):
                issues["missing_reasoning"] += 1
            if not evaluation.get("judge_model"):
                issues["missing_judge"] += 1

            # Check confidence
            confidence = evaluation.get("confidence")
            if confidence == "low":
                issues["low_confidence"] += 1
            elif confidence not in ["high", "medium", "low"]:
                issues["invalid_confidence"] += 1

    except Exception as e:
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
    answer_issues = scan_directory(Path("answers"), check_answer_issues)
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
