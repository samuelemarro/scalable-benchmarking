import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


from model_api import query_llm_single

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Get current timestamp in YYYYMMDD_HHMMSS format for versioning."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_json_with_metadata(path: Path, payload, add_timestamp: bool = True):
    """
    Save JSON with metadata including timestamp and version info.

    Args:
        path: Path to save the JSON file
        payload: Data to save (dict or list)
        add_timestamp: Whether to add generation timestamp to metadata
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # If payload is a list, wrap it with metadata
    if isinstance(payload, list):
        wrapped = {
            "data": payload,
            "metadata": {}
        }
    elif isinstance(payload, dict):
        # If already has metadata, use it; otherwise create new
        if "metadata" not in payload:
            wrapped = {
                **payload,
                "metadata": {}
            }
        else:
            wrapped = payload
    else:
        wrapped = {"data": payload, "metadata": {}}

    # Add timestamp if requested
    if add_timestamp and "generated_at" not in wrapped.get("metadata", {}):
        wrapped["metadata"]["generated_at"] = get_timestamp()

    with path.open("w") as f:
        json.dump(wrapped, f, indent=2)


def clean_math(text: str) -> str:
    """Clean LaTeX math delimiters by converting to $ and $$ formats."""
    text = text.replace("\\( ", "$").replace("\\(", "$")
    text = text.replace(" \\)", "$").replace("\\)", "$")
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    return text


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
        return None
    model = cfg.get("model")
    if not model:
        logger.warning("No model specified in parsing config")
        return None
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
        #repaired = clean_json_text(repaired)
        parsed = json.loads(repaired)

        if "parsing_error" in parsed:
            return None
        return parsed

    except Exception:
        logger.exception("Error repairing JSON with model")
        return None


def safe_load_json(text: str, schema_hint: Optional[str] = None) -> Optional[Dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    repaired = _repair_with_model(text, schema_hint)
    return repaired
