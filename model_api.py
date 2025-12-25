import json
import os
import re
import tempfile
import time
import uuid


import requests
from tqdm import tqdm
from google import genai
from google.genai import types as genai_types

OPENAI_API_URL = "https://api.openai.com/v1"
ANTHROPIC_INTERNAL_NAMES = {
    "claude-opus-4.1": "claude-opus-4-1",
    "claude-opus-4.5": "claude-opus-4-5-20251101",
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
    "claude-opus-4-5-20251101": 64000,
    "claude-opus-4": 32000,
    "claude-sonnet-4": 64000,
    "claude-3-7-sonnet": 64000,
    "claude-3-5-haiku": 8192,
    "claude-3-haiku": 4096,
}

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages/batches"
OPENAI_FILES_URL = f"{OPENAI_API_URL}/files"
OPENAI_BATCH_URL = f"{OPENAI_API_URL}/batches"
OPENAI_CONTENT_URL = f"{OPENAI_API_URL}/files/{{output_file_id}}/content"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

EFFORT_RATIOS = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2
} # OpenRouter effort to token ratio

def _validate_response_format(response_format):
    """Validate response_format parameter is in the correct format."""
    if response_format is None:
        return
    if not isinstance(response_format, dict):
        raise ValueError(f"response_format must be a dict, got {type(response_format).__name__}")
    if "type" not in response_format:
        raise ValueError("response_format dict must contain 'type' key")
    valid_types = {"json_object", "json_schema", "text"}
    if response_format["type"] not in valid_types:
        raise ValueError(f"response_format type must be one of {valid_types}, got '{response_format['type']}'")


# OpenRouter formula to compute max tokens from effort level
def anthropic_effort_to_tokens(model: str, effort: str):
    max_tokens = ANTHROPIC_MAX_TOKENS[model]

    return int(max(1024, min(32000, EFFORT_RATIOS[effort] * max_tokens)))


def _query_openai_single(model: str, messages: list, response_format: str, temperature: float, api_kwargs: dict, reasoning: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format
    if reasoning:
        payload["reasoning_effort"] = reasoning
    if api_kwargs:
        for k, v in api_kwargs.items():
            if k not in ["reasoning"]:
                payload[k] = v

    resp = requests.post(f"{OPENAI_API_URL}/chat/completions", headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI error: {resp.status_code} - {resp.text}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI error: No choices in response: {data}")
    message_obj = choices[0].get("message", {})
    content = message_obj.get("content")
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
        return "".join(text_parts)
    return content or ""


def query_llm(model: str, messages: list, response_format: str = None, temperature: float = 1, api_kwargs: dict = None, reasoning: str = None) -> str:
    """
    Query a single LLM endpoint (OpenRouter).
    """
    _validate_response_format(response_format)

    if api_kwargs:
        if api_kwargs.get("reasoning") is not None:
            raise ValueError("Do not set 'reasoning' in api_kwargs; use the reasoning parameter instead.")

        if api_kwargs.get("temperature") is not None:
            raise ValueError("Do not set 'temperature' in api_kwargs; use the temperature parameter instead.")

    if model.startswith("openai/"):
        return _query_openai_single(model.replace("openai/", ""), messages, response_format, temperature, api_kwargs, reasoning)

    if "gemini" in model:
        return _query_gemini_single(model, messages, response_format, temperature, api_kwargs, reasoning)

    if 'anthropic' in model and (temperature is not None and temperature != 1) and reasoning is not None:
        raise ValueError("Cannot set both temperature and reasoning in Anthropic requests")

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
        raise RuntimeError(f"OpenRouter error: No choices found in response: {data}")

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
            ]
        }

        if temperature is not None:
            body["temperature"] = temperature

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

        batch_requests.append((custom_id, body))

    return batch_requests, custom_ids


def _map_batch_results(results_text: str, custom_ids: list):
    """
    Map Anthropic .jsonl results to input order using custom_id.
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
            raise RuntimeError(f"Anthropic batch request failed: {result['type']}")

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
        raise RuntimeError(f"Anthropic error: Batch creation failed: {resp.status_code} - {resp.text}")
    batch_id = resp.json()["id"]
    poll_url = f"{ANTHROPIC_API_URL}/{batch_id}"

    # TQDM progress bar for polling
    with tqdm(total=1, desc="Polling Claude batch", bar_format='{desc}: {elapsed} [{bar}]') as pbar:
        while True:
            time.sleep(5)
            poll_resp = requests.get(poll_url, headers=headers)
            if poll_resp.status_code != 200:
                raise RuntimeError(f"Anthropic error: Batch poll failed: {poll_resp.status_code} - {poll_resp.text}")
            poll_data = poll_resp.json()
            status = poll_data.get("processing_status")

            if status == "ended":
                results_url = poll_data.get("results_url")
                pbar.update(1)
                break
            pbar.set_postfix_str(f"Status: {status}")

    results_resp = requests.get(results_url, headers=headers)
    if results_resp.status_code != 200:
        raise RuntimeError(f"Anthropic error: Batch results download failed: {results_resp.status_code} - {results_resp.text}")

    return _map_batch_results(results_resp.text, custom_ids)


def _build_openai_batch_requests(model: str, messages_list: list, prompt: str, response_format: str, temperature: float, api_kwargs: dict, reasoning: str = None):
    """
    Build batch requests for the OpenAI chat completions endpoint.
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
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            "temperature": temperature,
        }

        # OpenAI chat/completions batches do not accept the "reasoning" param; ignore if provided.
        if response_format:
            body["response_format"] = response_format
        if api_kwargs:
            for key, value in api_kwargs.items():
                if key != "reasoning":
                    body[key] = value

        batch_requests.append((custom_id, body))

    return batch_requests, custom_ids


def _map_openai_batch_results(results_text: str, custom_ids: list):
    """
    Map OpenAI .jsonl results to input order using custom_id.
    """
    responses_map = {}

    for line in results_text.strip().splitlines():
        result_obj = json.loads(line)
        cid = result_obj.get("custom_id")
        if not cid:
            continue

        response = result_obj.get("response")
        error = result_obj.get("error")

        if response and response.get("status_code") == 200:
            body = response.get("body", {})
            choices = body.get("choices") or []

            text_content = None
            if choices:
                message_obj = choices[0].get("message", {})
                content = message_obj.get("content")

                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
                    text_content = "".join(text_parts).strip() if text_parts else None
                else:
                    text_content = content

            if not text_content:
                raise RuntimeError(f"OpenAI batch request returned no text content for custom_id {cid}")
            responses_map[cid] = text_content
        else:
            raise RuntimeError(f"OpenAI batch request failed for custom_id {cid}: {error or 'Unknown error'}")

    if len(responses_map) != len(custom_ids):
        missing = [cid for cid in custom_ids if cid not in responses_map]
        raise RuntimeError(f"OpenAI batch missing responses for custom_ids: {missing}")

    return [responses_map[cid] for cid in custom_ids]


def _query_openai_batch(model: str, messages_list: list, prompt: str, response_format: str, temperature: float, api_kwargs: dict, reasoning: str = None):
    """
    Helper for OpenAI batch API using the chat completions endpoint.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    batch_requests, custom_ids = _build_openai_batch_requests(model, messages_list, prompt, response_format, temperature, api_kwargs, reasoning=reasoning)
    batch_lines = []

    for cid, body in batch_requests:
        batch_line = {
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {**body, "reasoning_effort": reasoning} if reasoning else body,
        }
        batch_lines.append(json.dumps(batch_line))

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as f:
        for line in batch_lines:
            f.write(line + "\n")
        batch_file_path = f.name

    headers_auth = {"Authorization": f"Bearer {api_key}"}
    try:
        with open(batch_file_path, "rb") as file_handle:
            files = {"file": (os.path.basename(batch_file_path), file_handle)}
            data = {"purpose": "batch"}
            upload_resp = requests.post(OPENAI_FILES_URL, headers=headers_auth, files=files, data=data)

        if upload_resp.status_code != 200:
            raise RuntimeError(f"OpenAI batch file upload failed: {upload_resp.status_code} - {upload_resp.text}")

        input_file_id = upload_resp.json()["id"]

        create_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        create_payload = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        create_resp = requests.post(OPENAI_BATCH_URL, headers=create_headers, json=create_payload)

        if create_resp.status_code != 200:
            raise RuntimeError(f"OpenAI batch creation failed: {create_resp.status_code} - {create_resp.text}")

        batch_id = create_resp.json()["id"]
        poll_url = f"{OPENAI_BATCH_URL}/{batch_id}"

        with tqdm(total=1, desc="Polling OpenAI batch", bar_format="{desc}: {elapsed} [{bar}]") as pbar:
            while True:
                poll_resp = requests.get(poll_url, headers=create_headers)
                if poll_resp.status_code != 200:
                    raise RuntimeError(f"OpenAI batch poll failed: {poll_resp.status_code} - {poll_resp.text}")

                poll_data = poll_resp.json()
                status = poll_data.get("status")

                if status == "completed":
                    output_file_id = poll_data.get("output_file_id")
                    if not output_file_id:
                        raise RuntimeError(f"OpenAI batch completed without output_file_id: {poll_data}")
                    pbar.update(1)
                    break
                if status in {"failed", "cancelled", "expired"}:
                    raise RuntimeError(f"OpenAI batch ended with status '{status}': {poll_data}")

                pbar.set_description(f"Polling OpenAI batch (status: {status})")
                time.sleep(5)

        content_url = OPENAI_CONTENT_URL.format(output_file_id=output_file_id)
        results_resp = requests.get(content_url, headers=headers_auth)
        if results_resp.status_code != 200:
            raise RuntimeError(f"OpenAI batch results download failed: {results_resp.status_code} - {results_resp.text}")

        return _map_openai_batch_results(results_resp.text, custom_ids)

    finally:
        if os.path.exists(batch_file_path):
            os.remove(batch_file_path)


def _single_query_worker(args):
    model, message, prompt, response_format, temperature, api_kwargs, reasoning = args
    return query_llm_single(model, message, prompt=prompt, response_format=response_format, temperature=temperature, api_kwargs=api_kwargs, reasoning=reasoning)


def _get_gemini_client():
    if not genai:
        raise RuntimeError("google-genai is not installed")
    api_key = os.getenv("GEMINI_API_KEY")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key, project=project)


def _normalize_gemini_model(model: str) -> str:
    return model.replace("google/", "")


def _gemini_contents_from_messages(messages: list, prompt: str):
    system_instruction = None
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "")
        if role == "system":
            system_instruction = text
        else:
            contents.append({"role": role, "parts": [{"text": text}]})
    if prompt and not system_instruction:
        system_instruction = prompt
    return contents, system_instruction


def _query_gemini_single(model: str, messages: list, response_format: str, temperature: float, api_kwargs: dict, reasoning: str):
    client = _get_gemini_client()
    model = _normalize_gemini_model(model)
    contents, system_instruction = _gemini_contents_from_messages(messages, prompt=None)
    thinking_config = None
    if reasoning in ["medium", "high"]:
        thinking_config = genai_types.ThinkingConfig(include_thoughts=False, thinking_level=reasoning)
    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        thinking_config=thinking_config,
        system_instruction=system_instruction,
    )
    #help(client.models.generate_content)
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
        #system_instruction=system_instruction,
    )
    parts = resp.candidates[0].content.parts if resp.candidates else []
    text_parts = [p.text for p in parts if getattr(p, "text", None)]
    return "".join(text_parts)


def _query_gemini_batch(model: str, messages_list: list, prompt: str, response_format: str, temperature: float, api_kwargs: dict, reasoning: str):
    client = _get_gemini_client()
    model = _normalize_gemini_model(model)
    requests_inline = []
    for msg in messages_list:
        contents, system_instruction = _gemini_contents_from_messages([{"role": "user", "content": msg}], prompt)
        requests_inline.append(
            genai_types.InlinedRequest(
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    thinking_config=genai_types.ThinkingConfig(
                        include_thoughts=False,
                        thinking_level=reasoning
                    ),
                    system_instruction=system_instruction
                )
            )
        )

    batch = client.batches.create(
        model=model,
        src=requests_inline
    )

    with tqdm(total=1, desc="Polling Gemini batch", bar_format="{desc}: {elapsed} [{bar}]") as pbar:
        while True:
            job = client.batches.get(name=batch.name)
            state = getattr(job, "state", None)
            state_name = state.name if state else "UNKNOWN"
            if state_name in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                if state_name != "JOB_STATE_SUCCEEDED":
                    raise RuntimeError(f"Gemini batch ended with state {state_name}")
                pbar.update(1)
                break
            pbar.set_postfix_str(f"State: {state_name}")
            time.sleep(5)

    if job.error:
        raise RuntimeError(f"Gemini batch error: {job.error}")
    responses = []

    if job.dest and getattr(job.dest, "file_name", None):
        file_name = job.dest.file_name
        content_bytes = client.files.download(file=file_name)
        text = content_bytes.decode("utf-8")
        for line in text.strip().splitlines():
            try:
                obj = json.loads(line)
                resp = obj.get("response") or obj.get("inlineResponse")
                if not resp:
                    raise RuntimeError("Gemini batch response missing response field")
                candidates = resp.get("candidates") or []
                if not candidates:
                    raise RuntimeError("Gemini batch response missing candidates")
                parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [p.get("text", "") for p in parts if p.get("text")]
                if not text_parts:
                    raise RuntimeError("Gemini batch response missing text content")
                responses.append("".join(text_parts))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse Gemini batch response: {e}")
    elif job.dest and getattr(job.dest, "inlined_responses", None):
        for inline_response in job.dest.inlined_responses:
            resp = getattr(inline_response, "response", None)
            if resp and getattr(resp, "candidates", None):
                parts = resp.candidates[0].content.parts
                text_parts = [p.text for p in parts if getattr(p, "text", None)]
                responses.append("".join(text_parts))
            elif getattr(inline_response, "error", None):
                raise RuntimeError(f"Gemini batch request failed: {inline_response.error}")
            else:
                raise RuntimeError("Gemini batch request returned no response")
    else:
        raise RuntimeError("Gemini batch job has no recognized destination format")

    if len(responses) != len(messages_list):
        raise RuntimeError(f"Gemini batch returned {len(responses)} responses but expected {len(messages_list)}")
    return responses


def query_llm_batch(model: str, messages_list: list, prompt: str = "You are a helpful assistant.", response_format: str = None, temperature: float = 1, api_kwargs: dict = None, reasoning: str = None, max_workers: int = 8) -> list:
    """
    Batch query for LLMs: only Anthropic Claude batch API is used. All other models use parallel processing (multiprocessing).
    reasoning: None, "medium", or "high". If set, enables Claude 'thinking' or OpenRouter 'reasoning'.
    max_workers: number of parallel workers for non-Anthropic models.
    """
    if not messages_list:
        return []

    _validate_response_format(response_format)

    if api_kwargs:
        if api_kwargs.get("reasoning") is not None:
            raise ValueError("Do not set 'reasoning' in api_kwargs; use the reasoning parameter instead.")

        if api_kwargs.get("temperature") is not None:
            raise ValueError("Do not set 'temperature' in api_kwargs; use the temperature parameter instead.")

    if "gemini" in model:
        return _query_gemini_batch(model, messages_list, prompt, response_format, temperature, api_kwargs, reasoning)
    if 'anthropic' in model:
        if (temperature is not None and temperature != 1) and reasoning is not None:
            raise ValueError("Cannot set both temperature and reasoning in Anthropic requests")
        norm_model = ANTHROPIC_INTERNAL_NAMES.get(model.replace('anthropic/', ''), model.replace('anthropic/', ''))
        return _query_anthropic_batch(norm_model, messages_list, prompt, response_format, temperature, api_kwargs, reasoning=reasoning)
    if 'openai' in model:
        return _query_openai_batch(model.replace('openai/', ''), messages_list, prompt, response_format, temperature, api_kwargs, reasoning=reasoning)
    else:
        from multiprocessing import Pool
        worker_args = [(model, message, prompt, response_format, temperature, api_kwargs, reasoning) for message in messages_list]
        with Pool(processes=max_workers) as pool:
            results = pool.map(_single_query_worker, worker_args)
        return results
