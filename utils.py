import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional


from model_api import query_llm_single
from data_models import AnswerAttempt, AnswerEntry, BenchmarkEntry

logger = logging.getLogger(__name__)


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


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


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


def _repair_with_model(text: str, schema_hint: Optional[str]) -> Optional[Dict]:
    cfg = _load_parsing_config()

    if not cfg:
        raise RuntimeError("No parsing config found for JSON repair")
    model = cfg.get("model")
    if not model:
        raise RuntimeError("No model specified in parsing config for JSON repair")
    temperature = cfg.get("temperature", None)
    prompt_lines = [
        "You are a strict JSON repair assistant.",
        "Given the malformed text below, output a valid JSON object only.",
        "Be careful to fix stuff like incorrect escaping, missing commas, unbalanced braces, and incorrect quotes.",
        "Some escapings might be done correctly, so be cautious not to over-escape. Use contextual judgment.",
        "You will handle what is mostly math, so use your knowledge of LaTeX syntax to avoid breaking math expressions.",
        "Do not add, remove or change the content of any fields.",
    ]
    if schema_hint:
        prompt_lines.append(f"Schema: {schema_hint}")
    prompt_lines.append('If you cannot repair, respond with the following JSON: { \"parsing_error\" : \"Cannot parse\" }.')
    prompt = "\n".join(prompt_lines)

    try:
        repaired = query_llm_single(model, text, prompt=prompt, temperature=temperature, response_format={"type": "json_object"})
        print('Repaired type:', type(repaired))
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
            return None
        return parsed

    except Exception:
        logger.exception("Error repairing JSON with model")
        return None


def safe_load_json(text: str, schema_hint: Optional[str] = None, strict: bool = False) -> Optional[Dict]:
    """
    Attempt to parse JSON with progressive fallback strategies.

    Args:
        text: The text to parse as JSON
        schema_hint: Optional hint about expected schema for repair
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
    repaired = _repair_with_model(text, schema_hint)
    if repaired:
        logger.info("JSON repaired successfully with LLM")
    else:
        logger.warning("Failed to repair JSON even with LLM assistance")
    return repaired
