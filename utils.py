import json
from pathlib import Path
from typing import Dict, Optional


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


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


def _repair_with_model(text: str, schema_hint: Optional[str]) -> Optional[dict]:
    cfg = _load_parsing_config()
    if not cfg:
        return None
    model = cfg.get("model")
    if not model:
        return None
    prompt_lines = [
        "You are a strict JSON repair assistant.",
        "Given the malformed text below, output a valid JSON object only.",
    ]
    if schema_hint:
        prompt_lines.append(f"Schema: {schema_hint}")
    prompt_lines.append('If you cannot repair, respond with the string "can\'t parse".')
    prompt = "\n".join(prompt_lines)
    try:
        from model_api import query_llm_single

        repaired = query_llm_single(model, text, prompt=prompt, temperature=0)
        repaired = clean_json_text(repaired)
        return json.loads(repaired)
    except Exception:
        return None


def safe_load_json(text: str, schema_hint: Optional[str] = None) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    repaired = _repair_with_model(cleaned, schema_hint)
    return repaired
