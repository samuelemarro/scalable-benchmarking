from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from model_api import query_llm_batch, query_llm_single
from utils import safe_load_json


@dataclass
class Attempt:
    round: int
    answer: str
    evaluation: Optional[Dict] = None
    improved_from: Optional[int] = None


@dataclass
class ImprovementResult:
    final_answer: Optional[str]
    attempts: List[Attempt] = field(default_factory=list)
    status: str = "failed"  # succeeded/failed
    last_feedback: Optional[Dict] = None


def _batched_query(model: str, prompts: Sequence[str], disable_batch: bool, **kwargs) -> List[str]:
    if len(prompts) == 1 or disable_batch:
        return [query_llm_single(model, prompts[0], **kwargs)]
    return query_llm_batch(model, prompts, **kwargs)


def self_improve_answers(
    model: str,
    questions: Sequence[str],
    initial_answers: Sequence[str],
    build_eval_prompt: Callable[[str, str, int], str],
    build_refine_prompt: Callable[[str, str, str], str],
    max_rounds: int = 3,
    disable_batch: bool = False,
    temperature: float = 0.7,
    reasoning: Optional[str] = None,
) -> List[ImprovementResult]:
    """
    Perform self-critique/improvement loops on a batch of answers.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    results = [ImprovementResult(final_answer=ans) for ans in initial_answers]
    active_indices = list(range(len(questions)))

    for round_idx in range(max_rounds):
        eval_prompts = [
            build_eval_prompt(questions[i], results[i].final_answer, i) for i in active_indices
        ]
        eval_responses = _batched_query(
            model,
            eval_prompts,
            disable_batch,
            temperature=temperature,
            reasoning=reasoning,
        )

        next_active = []
        refine_prompts = []
        refine_indices = []

        for idx, eval_text in zip(active_indices, eval_responses):
            evaluation = safe_load_json(
                eval_text,
                schema_hint='{"verdict": "pass|fail", "ill_posed": bool, "issues": [string], "improvements": string}',
            ) or {
                "verdict": "fail",
                "issues": ["Could not parse evaluation JSON."],
                "ill_posed": False,
                "improvements": "Rewrite the answer carefully.",
            }
            attempt = Attempt(
                round=round_idx + 1,
                answer=results[idx].final_answer,
                evaluation=evaluation,
                improved_from=round_idx,
            )
            results[idx].attempts.append(attempt)
            results[idx].last_feedback = evaluation

            if evaluation.get("verdict") == "pass":
                results[idx].status = "succeeded"
                continue

            if evaluation.get("ill_posed"):
                results[idx].status = "failed"
                continue

            if round_idx == max_rounds - 1:
                results[idx].status = "failed"
                continue

            refine_indices.append(idx)
            refine_prompts.append(
                build_refine_prompt(
                    questions[idx],
                    results[idx].final_answer,
                    evaluation.get("improvements", "Improve correctness and completeness."),
                )
            )
            next_active.append(idx)

        if not refine_prompts:
            break

        refined_answers = _batched_query(
            model,
            refine_prompts,
            disable_batch,
            temperature=temperature,
            reasoning=reasoning,
        )
        for idx, new_answer in zip(refine_indices, refined_answers):
            results[idx].final_answer = new_answer

        active_indices = next_active

    return results
