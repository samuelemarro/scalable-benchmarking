import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


from model_api import query_llm_single
from data_models import AnswerAttempt, AnswerEntry, BenchmarkEntry

logger = logging.getLogger(__name__)

QuestionKey = Tuple[Optional[str], Optional[str], Optional[str]]
AnswerKey = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]
CritiqueKey = Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]
EvaluationKey = Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
]


def _normalize_part(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def question_key(
    question_model: Optional[str], run_id: Optional[str], outer_attempt: Optional[str]
) -> Optional[QuestionKey]:
    if run_id is None or not question_model or outer_attempt is None:
        return None
    return (_normalize_part(run_id), _normalize_part(question_model), _normalize_part(outer_attempt))


def question_key_from_entry(entry: Any, question_model: Optional[str]) -> Optional[QuestionKey]:
    if not entry:
        return None
    run_id = entry.get("run_id") if isinstance(entry, dict) else getattr(entry, "run_id", None)
    outer_attempt = entry.get("outer_attempt") if isinstance(entry, dict) else getattr(entry, "outer_attempt", None)
    return question_key(question_model, run_id, outer_attempt)


def answer_key(
    question_model: Optional[str],
    answer_model: Optional[str],
    run_id: Optional[str],
    outer_attempt: Optional[str],
) -> Optional[AnswerKey]:
    if run_id is None or not question_model or not answer_model or outer_attempt is None:
        return None
    return (
        _normalize_part(run_id),
        _normalize_part(question_model),
        _normalize_part(answer_model),
        _normalize_part(outer_attempt),
    )


def answer_key_from_entry(entry: Any) -> Optional[AnswerKey]:
    if not entry:
        return None
    if isinstance(entry, dict):
        return answer_key(
            entry.get("question_model"),
            entry.get("answer_model"),
            entry.get("run_id"),
            entry.get("outer_attempt"),
        )
    return answer_key(
        getattr(entry, "question_model", None),
        getattr(entry, "answer_model", None),
        getattr(entry, "run_id", None),
        getattr(entry, "outer_attempt", None),
    )


def critique_key(
    question_model: Optional[str],
    answer_model: Optional[str],
    critic_model: Optional[str],
    mode: Optional[str],
    run_id: Optional[str],
    outer_attempt: Optional[str],
) -> Optional[CritiqueKey]:
    if run_id is None or not question_model or not answer_model or outer_attempt is None:
        return None
    return (
        _normalize_part(run_id),
        _normalize_part(question_model),
        _normalize_part(answer_model),
        _normalize_part(critic_model),
        _normalize_part(mode),
        _normalize_part(outer_attempt),
    )


def critique_key_from_entry(entry: Any, mode: Optional[str]) -> Optional[CritiqueKey]:
    if not entry:
        return None
    if isinstance(entry, dict):
        return critique_key(
            entry.get("question_author"),
            entry.get("answer_author"),
            entry.get("critic"),
            mode,
            entry.get("run_id"),
            entry.get("outer_attempt"),
        )
    return critique_key(
        getattr(entry, "question_author", None),
        getattr(entry, "answer_author", None),
        getattr(entry, "critic", None),
        mode,
        getattr(entry, "run_id", None),
        getattr(entry, "outer_attempt", None),
    )


def debate_key(
    question_model: Optional[str],
    answer_model: Optional[str],
    critic_model: Optional[str],
    mode: Optional[str],
    run_id: Optional[str],
    outer_attempt: Optional[str],
) -> Optional[CritiqueKey]:
    return critique_key(question_model, answer_model, critic_model, mode, run_id, outer_attempt)


def debate_key_from_entry(
    entry: Any,
    question_model: Optional[str],
    answer_model: Optional[str],
    critic_model: Optional[str],
    mode: Optional[str],
) -> Optional[CritiqueKey]:
    if not entry:
        return None
    run_id = entry.get("run_id") if isinstance(entry, dict) else getattr(entry, "run_id", None)
    outer_attempt = entry.get("outer_attempt") if isinstance(entry, dict) else getattr(entry, "outer_attempt", None)
    return debate_key(question_model, answer_model, critic_model, mode, run_id, outer_attempt)


def automated_evaluation_key_from_entry(entry: Any) -> Optional[EvaluationKey]:
    if not entry:
        return None
    if isinstance(entry, dict):
        return (
            _normalize_part(entry.get("run_id")),
            _normalize_part(entry.get("question_model")),
            _normalize_part(entry.get("answer_model")),
            _normalize_part(entry.get("critic_model")),
            _normalize_part(entry.get("mode")),
            _normalize_part(entry.get("judge_model")),
            _normalize_part(entry.get("outer_attempt")),
        )
    return (
        _normalize_part(getattr(entry, "run_id", None)),
        _normalize_part(getattr(entry, "question_model", None)),
        _normalize_part(getattr(entry, "answer_model", None)),
        _normalize_part(getattr(entry, "critic_model", None)),
        _normalize_part(getattr(entry, "mode", None)),
        _normalize_part(getattr(entry, "judge_model", None)),
        _normalize_part(getattr(entry, "outer_attempt", None)),
    )


def judging_task_key(entry: Any) -> Optional[CritiqueKey]:
    if not entry:
        return None
    if isinstance(entry, dict):
        return critique_key(
            entry.get("question_model"),
            entry.get("answer_model"),
            entry.get("critic_model"),
            entry.get("mode"),
            entry.get("run_id"),
            entry.get("outer_attempt"),
        )
    return critique_key(
        getattr(entry, "question_model", None),
        getattr(entry, "answer_model", None),
        getattr(entry, "critic_model", None),
        getattr(entry, "mode", None),
        getattr(entry, "run_id", None),
        getattr(entry, "outer_attempt", None),
    )


def automated_evaluation_key_for_task(entry: Any, judge_model: Optional[str]) -> Optional[EvaluationKey]:
    base = judging_task_key(entry)
    if not base:
        return None
    run_id, question_model, answer_model, critic_model, mode, outer_attempt = base
    return (
        _normalize_part(run_id),
        _normalize_part(question_model),
        _normalize_part(answer_model),
        _normalize_part(critic_model),
        _normalize_part(mode),
        _normalize_part(judge_model),
        _normalize_part(outer_attempt),
    )


def human_evaluation_key_from_entry(entry: Any) -> Optional[CritiqueKey]:
    return judging_task_key(entry)


def format_key(parts: Sequence[Optional[str]]) -> str:
    return "/".join("unknown" if part in (None, "") else str(part) for part in parts)


def task_key_from_prefix(prefix: str, run_id: Optional[str], outer_attempt: Optional[str]) -> Optional[CritiqueKey]:
    if run_id is None or outer_attempt is None:
        return None
    if not prefix:
        return None
    parts = prefix.split("/")
    if not parts:
        return None
    if parts[0] == "illposed":
        if len(parts) < 3:
            return None
        return critique_key(parts[1], parts[2], None, None, run_id, outer_attempt)
    if parts[0] == "critique":
        if len(parts) < 4:
            return None
        mode = parts[1]
        question_model = parts[2]
        pair = parts[3]
        if "__" not in pair:
            return None
        critic_model, answer_model = pair.split("__", 1)
        return critique_key(question_model, answer_model, critic_model, mode, run_id, outer_attempt)
    return None


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    # Disable httpx logging at INFO level to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _ensure_non_empty_responses(responses: Sequence[Optional[str]], context: str) -> None:
    for idx, response in enumerate(responses):
        if response is None or (isinstance(response, str) and not response.strip()):
            raise ValueError(f"{context} at index {idx}")


def clean_math(text: str) -> str:
    """Clean LaTeX math delimiters by converting to $ and $$ formats."""
    if not text:
        return text
    text = text.replace("\\( ", "$").replace("\\(", "$")
    text = text.replace(" \\)", "$").replace("\\)", "$")
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    for env in ("equation", "equation*", "align", "align*", "gather", "gather*"):
        text = text.replace(f"\\begin{{{env}}}", "$$")
        text = text.replace(f"\\end{{{env}}}", "$$")
    return text


def benchmark_answers_from_entries(
    question_model_slug: str,
    benchmark_entries: List[Optional[BenchmarkEntry]],
) -> List[Optional[AnswerEntry]]:
    """Build AnswerEntry records from benchmark entries when self-answer files are disallowed or missing."""
    answers: List[Optional[AnswerEntry]] = []
    for entry in benchmark_entries:
        if not entry:
            answers.append(None)
            continue
        gen_rounds = entry.generation_rounds or []
        if not gen_rounds:
            answers.append(None)
            continue
        refinements = gen_rounds[-1].refinement_rounds or []
        if not refinements:
            answers.append(None)
            continue
        last_ref = refinements[-1]
        attempts = [
            AnswerAttempt(
                round=att.round,
                answer=att.answer,
                raw_answer=att.raw_answer,
                evaluation=att.evaluation,
            )
            for att in refinements
        ]
        answers.append(
            AnswerEntry(
                question_model=question_model_slug,
                answer_model=question_model_slug,
                question=last_ref.question,
                run_id=entry.run_id,
                outer_attempt=entry.outer_attempt,
                topic_slug=entry.topic_slug,
                status=entry.status,
                attempts=attempts,
            )
        )
    return answers


def _load_parsing_config() -> Optional[Dict]:
    cfg_path = Path("configs/parsing.json")
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            return None
    return None


def clean_json_text(text: str) -> str:
    text = text.replace("```json", "```")

    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            # take first fenced block
            return parts[1].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _repair_with_model(text: str, schema: Optional[Dict[str, Any]]) -> Optional[Dict]:
    cfg = _load_parsing_config()

    if not cfg:
        raise RuntimeError("No parsing config found for JSON repair")
    model = cfg.get("model")
    if not model:
        raise RuntimeError("No model specified in parsing config for JSON repair")
    temperature = cfg.get("temperature", None)
    reasoning = cfg.get("reasoning", None)
    prompt_lines = [
        "You are a strict JSON repair assistant.",
        "Given the malformed text below, output a valid JSON object only.",
        "Be careful to fix stuff like incorrect escaping, missing commas, unbalanced braces, and incorrect quotes.",
        "Some escapings might be done correctly, so be cautious not to over-escape. Use contextual judgment.",
        "You will handle what is mostly math, so use your knowledge of LaTeX syntax to avoid breaking math expressions.",
        "Do not add, remove or change the content of any fields. In particular, if a string field doesn't match an enum allowed value, fail to parse rather than changing it.",
        "Note: if a field is optional in the schema, but the value is null in the input, it should be dropped in the output (unless null is specifically allowed by the schema).",
        "If there are extra fields not in the schema, drop them.",
        "Note 2: If the input text is natural language instead of JSON, but it contains all the relevant information in sufficiently obvious format, try to convert it into JSON format. Just don't try to invent fields.",
    ]
    if schema:
        prompt_lines.append("JSON Schema:")
        prompt_lines.append(json.dumps(schema, indent=2, sort_keys=False))
    prompt_lines.append('If you cannot repair, respond with the following JSON: { \"parsing_error\" : \"Cannot parse\", \"reason\": ... }.')
    prompt = "\n".join(prompt_lines)
    prompt += "\n\nMalformed text:\n"

    for i in range(3): # Up to 3 attempts because we live in a world where determinism is dead
        try:
            repaired = query_llm_single(
                model,
                text,
                prompt=prompt,
                temperature=temperature,
                response_format={"type": "json_object"},
                reasoning=reasoning,
            )
            #repaired = clean_json_text(repaired)
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                # Last attempt: clean and try again
                cleaned = clean_json_text(repaired)

                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    parsed = {"parsing_error": "Cannot parse"}

            if "parsing_error" in parsed:
                logger.warning(f"LLM JSON repair attempt {i+1} failed: {parsed.get('reason', 'no reason provided')}")
            else:
                return parsed

        except Exception:
            logger.exception("Error repairing JSON with model")

    return None


def safe_load_json(text: str, schema: Optional[Dict[str, Any]] = None, strict: bool = False) -> Optional[Dict]:
    """
    Attempt to parse JSON with progressive fallback strategies.

    Args:
        text: The text to parse as JSON
        schema: Optional JSON Schema for repair
        strict: If True, only allow direct JSON parsing (no cleaning or repair)

    Returns:
        Parsed JSON dict, or None if parsing failed
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if strict:
            logger.warning("Strict mode: Failed to parse JSON directly, not attempting fallback")
            return None

    # Fallback 1: Try cleaning common issues
    cleaned = clean_json_text(text)
    try:
        parsed = json.loads(cleaned)
        logger.debug("JSON parsed successfully after cleaning")
        return parsed
    except json.JSONDecodeError:
        pass

    # Fallback 2: Use LLM to repair
    logger.info("Attempting LLM-based JSON repair")
    repaired = _repair_with_model(text, schema)
    if repaired:
        logger.info("JSON repaired successfully with LLM")
    else:
        logger.warning("Failed to repair JSON even with LLM assistance")
    return repaired
