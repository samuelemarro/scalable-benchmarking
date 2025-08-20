import os
import requests

def query_llm(model, messages, response_format=None, temperature=1, api_kwargs=None):
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

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=json_kwargs,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Error: {response.status_code} - {response.text}")
    data = response.json()

    if "choices" not in data or not data["choices"]:
        raise RuntimeError("No choices found in response")
    return data["choices"][0]["message"]["content"]

def query_llm_single(model, message, prompt="You are a helpful assistant.", response_format=None, temperature=1, api_kwargs=None):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message},
    ]
    return query_llm(model, messages, response_format=response_format, temperature=temperature, api_kwargs=api_kwargs)