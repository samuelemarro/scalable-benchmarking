import json
import os
import tempfile
import time
import uuid

import requests
from tqdm import tqdm

ANTHROPIC_INTERNAL_NAMES = {
    "claude-opus-4.1": "claude-opus-4-1",
    "claude-opus-4": "claude-opus-4",
    "claude-sonnet-4": "claude-sonnet-4",
    "claude-3.7-sonnet": "claude-3-7-sonnet",
    "claude-sonnet-3.7": "claude-3-7-sonnet",
    "claude-3.5-haiku": "claude-3-5-haiku",
    "claude-haiku-3.5": "claude-3-5-haiku",
    "claude-3-haiku": "claude-3-haiku",
    "claude-haiku-3": "claude-3-haiku"
}

ANTHROPIC_MAX_TOKENS = {
    "claude-opus-4-1": 32000,
    "claude-opus-4": 32000,
    "claude-sonnet-4": 64000,
    "claude-3-7-sonnet": 64000,
    "claude-3-5-haiku": 8192,
    "claude-3-haiku": 4096,
}

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages/batches"
OPENAI_FILES_URL = "https://api.openai.com/v1/files"
OPENAI_BATCH_URL = "https://api.openai.com/v1/batches"
OPENAI_CONTENT_URL = "https://api.openai.com/v1/files/{output_file_id}/content"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

EFFORT_RATIOS = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2
} # OpenRouter effort to token ratio

# OpenRouter formula to compute max tokens from effort level
def anthropic_effort_to_tokens(model: str, effort: str):
    max_tokens = ANTHROPIC_MAX_TOKENS[model]

    return int(max(1024, min(32000, EFFORT_RATIOS[effort] * max_tokens)))


def query_llm(model: str, messages: list, response_format: str = None, temperature: float = 1, api_kwargs: dict = None, reasoning: str = None) -> str:
    """
    Query a single LLM endpoint (OpenRouter).
    """

    if 'anthropic' in model and api_kwargs and api_kwargs.get("thinking") and temperature != 1:
        raise ValueError("Cannot set both temperature and thinking in Anthropic requests")

    json_kwargs = {}

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    json_kwargs["model"] = model
    json_kwargs["messages"] = messages
    json_kwargs["temperature"] = temperature

    if response_format:
        json_kwargs["response_format"] = response_format

    # Add reasoning to api_kwargs for OpenRouter
    if reasoning in ["medium", "high"]:
        json_kwargs["reasoning"] = {"effort": reasoning, "exclude": True}
    if api_kwargs:
        for key, value in api_kwargs.items():
            json_kwargs[key] = value

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=json_kwargs)

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error: {response.status_code} - {response.text}")

    data = response.json()

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"No choices found in response: {data}")

    return data["choices"][0]["message"]["content"]


def query_llm_single(model, message, prompt="You are a helpful assistant.", response_format=None, temperature=1, api_kwargs=None, reasoning=None):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message},
    ]
    return query_llm(model, messages, response_format=response_format, temperature=temperature, api_kwargs=api_kwargs, reasoning=reasoning)


def _build_batch_requests(model: str, messages_list: list, prompt: str, response_format: str, temperature: float, api_kwargs: dict, reasoning: str = None):
    """
    Build batch requests for the Anthropic Claude API.
    Returns: (batch_requests, custom_ids)
    """
    batch_requests = []
    custom_ids = []

    for message in messages_list:
        custom_id = f"msg-{uuid.uuid4()}"
        custom_ids.append(custom_id)

        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": message},
            ],
            "temperature": temperature
        }
        if prompt:
            body["system"] = prompt
        norm_model = ANTHROPIC_INTERNAL_NAMES.get(model, model)
        max_tokens = ANTHROPIC_MAX_TOKENS.get(norm_model)
        if max_tokens is None:
            for k in ANTHROPIC_MAX_TOKENS:
                if norm_model.startswith(k):
                    max_tokens = ANTHROPIC_MAX_TOKENS[k]
                    break
        if max_tokens:
            body["max_tokens"] = max_tokens
        else:
            raise ValueError(f"Model '{model}' not found in ANTHROPIC_MAX_TOKENS")
        # Enable thinking if reasoning is set
        if reasoning in ["medium", "high"]:
            effort = reasoning
            thinking_tokens = anthropic_effort_to_tokens(norm_model, effort)
            body["thinking"] = {"type": "enabled", "budget_tokens": thinking_tokens}
        if response_format:
            body["response_format"] = response_format
        if api_kwargs:
            for key, value in api_kwargs.items():
                if key not in ["reasoning", "thinking"]:
                    body[key] = value

        print('\n' * 3)
        print('body:', body)

        batch_requests.append((custom_id, body))

    return batch_requests, custom_ids


def _map_batch_results(results_text: str, custom_ids: list):
    """
    Map .jsonl results to input order using custom_id.
    """
    responses_map = {}

    for line in results_text.strip().splitlines():
        result_obj = json.loads(line)
        cid = result_obj["custom_id"]
        result = result_obj["result"]
        if result["type"] == "succeeded":
            sub_result = result["message"]["content"]
            
            # Find the first result with type 'text'
            for item in sub_result:
                if item["type"] == "text":
                    responses_map[cid] = item["text"]
                    break
            else:
                raise RuntimeError(f"No text content found in Anthropic response: {result}")
        else:
            responses_map[cid] = f"[Error: {result['type']}]"

    return [responses_map[cid] for cid in custom_ids]


def _query_anthropic_batch(model: str, messages_list: list, prompt: str, response_format: str, temperature: float, api_kwargs: dict, reasoning: str = None):
    """
    Helper for Anthropic Claude batch API.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    batch_requests, custom_ids = _build_batch_requests(model, messages_list, prompt, response_format, temperature, api_kwargs, reasoning=reasoning)
    batch_payload = {"requests": [
        {"custom_id": cid, "params": body} for cid, body in batch_requests
    ]}
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=batch_payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Claude batch creation failed: {resp.status_code} - {resp.text}")
    batch_id = resp.json()["id"]
    poll_url = f"{ANTHROPIC_API_URL}/{batch_id}"

    # TQDM progress bar for polling
    with tqdm(total=1, desc="Polling Claude batch", bar_format='{desc}: {elapsed} [{bar}]') as pbar:
        while True:
            time.sleep(5)
            poll_resp = requests.get(poll_url, headers=headers)
            if poll_resp.status_code != 200:
                raise RuntimeError(f"Claude batch poll failed: {poll_resp.status_code} - {poll_resp.text}")
            poll_data = poll_resp.json()
            status = poll_data.get("processing_status")

            if status == "ended":
                results_url = poll_data.get("results_url")
                pbar.update(1)
                break
            pbar.set_postfix_str(f"Status: {status}")

    results_resp = requests.get(results_url, headers=headers)
    if results_resp.status_code != 200:
        raise RuntimeError(f"Claude batch results download failed: {results_resp.status_code} - {results_resp.text}")

    return _map_batch_results(results_resp.text, custom_ids)


def query_llm_batch(model: str, messages_list: list, prompt: str = "You are a helpful assistant.", response_format: str = None, temperature: float = 1, api_kwargs: dict = None, reasoning: str = None) -> list:
    """
    Batch query for LLMs: only Anthropic Claude batch API is used. All other models use classic loop.
    reasoning: None, "medium", or "high". If set, enables Claude 'thinking' or OpenRouter 'reasoning'.
    """
    if 'anthropic' in model:
        if temperature != 1 and (reasoning in ["medium", "high"]):
            raise ValueError("Cannot set both temperature and reasoning in Anthropic requests")
        norm_model = ANTHROPIC_INTERNAL_NAMES.get(model.replace('anthropic/', ''), model.replace('anthropic/', ''))
        return _query_anthropic_batch(norm_model, messages_list, prompt, response_format, temperature, api_kwargs, reasoning=reasoning)
    else:
        results = []
        for message in messages_list:
            result = query_llm_single(model, message, prompt=prompt, response_format=response_format, temperature=temperature, api_kwargs=api_kwargs, reasoning=reasoning)
            results.append(result)
        return results