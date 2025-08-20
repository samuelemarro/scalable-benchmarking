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


def query_llm(model: str, messages: list, response_format: str = None, temperature: float = 1, api_kwargs: dict = None) -> str:
    """
    Query a single LLM endpoint (OpenRouter).
    """
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


def query_llm_single(model, message, prompt="You are a helpful assistant.", response_format=None, temperature=1, api_kwargs=None):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message},
    ]
    return query_llm(model, messages, response_format=response_format, temperature=temperature, api_kwargs=api_kwargs)


def _build_batch_requests(model: str, messages_list: list, prompt: str, response_format: str, temperature: float, api_kwargs: dict, is_anthropic: bool = False):
    """
    Build batch requests for both OpenAI and Anthropic APIs.
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

        if is_anthropic:
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
        else:
            body["messages"].insert(0, {"role": "system", "content": prompt})

        if response_format:
            body["response_format"] = response_format

        if api_kwargs:
            for key, value in api_kwargs.items():
                body[key] = value

        batch_requests.append((custom_id, body))

    return batch_requests, custom_ids


def _map_batch_results(results_text: str, custom_ids: list, result_type: str):
    """
    Map .jsonl results to input order using custom_id.
    result_type: 'anthropic' or 'openai'
    """
    responses_map = {}

    for line in results_text.strip().splitlines():
        result_obj = json.loads(line)
        cid = result_obj["custom_id"]
        if result_type == "anthropic":
            result = result_obj["result"]
            if result["type"] == "succeeded":
                responses_map[cid] = result["message"]["content"][0]["text"]
            else:
                responses_map[cid] = f"[Error: {result['type']}]"
        else:  # openai
            response = result_obj.get("response")
            error = result_obj.get("error")
            if response and response.get("status_code") == 200:
                body = response["body"]
                choices = body.get("choices", [])
                if choices:
                    responses_map[cid] = choices[0]["message"]["content"]
                else:
                    responses_map[cid] = "[Error: No choices]"
            else:
                responses_map[cid] = f"[Error: {error}]"

    return [responses_map[cid] for cid in custom_ids]


def _query_anthropic_batch(model: str, messages_list: list, prompt: str, response_format: str, temperature: float, api_kwargs: dict):
    """
    Helper for Anthropic Claude batch API.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    batch_requests, custom_ids = _build_batch_requests(model, messages_list, prompt, response_format, temperature, api_kwargs, is_anthropic=True)
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

    return _map_batch_results(results_resp.text, custom_ids, "anthropic")


def _query_openai_batch(model: str, messages_list: list, prompt: str, response_format: str, temperature: float, api_kwargs: dict):
    """
    Helper for OpenAI Batch API.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    batch_requests, custom_ids = _build_batch_requests(model, messages_list, prompt, response_format, temperature, api_kwargs)
    batch_lines = []

    for cid, body in batch_requests:
        batch_line = {
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        batch_lines.append(json.dumps(batch_line))

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as f:
        for line in batch_lines:
            f.write(line + "\n")
        batch_file_path = f.name

    files_headers = {
        "Authorization": f"Bearer {openai_api_key}"
    }
    files_data = {
        "purpose": "batch"
    }
    with open(batch_file_path, "rb") as file_data:
        files_resp = requests.post(OPENAI_FILES_URL, headers=files_headers, data=files_data, files={"file": file_data})
    if files_resp.status_code != 200:
        raise RuntimeError(f"OpenAI batch file upload failed: {files_resp.status_code} - {files_resp.text}")
    input_file_id = files_resp.json()["id"]
    batch_headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    batch_payload = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    }
    batch_resp = requests.post(OPENAI_BATCH_URL, headers=batch_headers, json=batch_payload)
    if batch_resp.status_code != 200:
        raise RuntimeError(f"OpenAI batch creation failed: {batch_resp.status_code} - {batch_resp.text}")
    batch_id = batch_resp.json()["id"]
    poll_url = f"{OPENAI_BATCH_URL}/{batch_id}"

    # TQDM progress bar for polling
    with tqdm(total=1, desc="Polling OpenAI batch", bar_format='{desc}: {elapsed} [{bar}]') as pbar:
        while True:
            poll_resp = requests.get(poll_url, headers=batch_headers)
            if poll_resp.status_code != 200:
                raise RuntimeError(f"OpenAI batch poll failed: {poll_resp.status_code} - {poll_resp.text}")
            poll_data = poll_resp.json()
            status = poll_data.get("status")
            if status == "completed":
                output_file_id = poll_data.get("output_file_id")
                pbar.update(1)
                break
            elif status == "failed":
                raise RuntimeError(f"OpenAI batch failed: {poll_data}")
            pbar.set_postfix_str(f"Status: {status}")
            time.sleep(5)

    results_url = OPENAI_CONTENT_URL.format(output_file_id=output_file_id)
    results_resp = requests.get(results_url, headers=files_headers)
    if results_resp.status_code != 200:
        raise RuntimeError(f"OpenAI batch results download failed: {results_resp.status_code} - {results_resp.text}")

    return _map_batch_results(results_resp.text, custom_ids, "openai")


def query_llm_batch(model: str, messages_list: list, prompt: str = "You are a helpful assistant.", response_format: str = None, temperature: float = 1, api_kwargs: dict = None) -> list:
    """
    Batch query for LLMs (Anthropic, OpenAI, or fallback).
    """
    if 'anthropic' in model:
        norm_model = ANTHROPIC_INTERNAL_NAMES.get(model.replace('anthropic/', ''), model.replace('anthropic/', ''))
        return _query_anthropic_batch(norm_model, messages_list, prompt, response_format, temperature, api_kwargs)
    elif 'openai' in model:
        return _query_openai_batch(model.replace('openai/', ''), messages_list, prompt, response_format, temperature, api_kwargs)
    else:
        results = []
        for message in messages_list:
            result = query_llm_single(model, message, prompt=prompt, response_format=response_format, temperature=temperature, api_kwargs=api_kwargs)
            results.append(result)
        return results