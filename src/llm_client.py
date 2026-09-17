"""
llm_client.py

Single point of contact with the LLM API. Uses Groq's free tier.
"""

import os
import time
import requests

AGENT_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"

BASE_URL = "https://api.groq.com/openai/v1/chat/completions"


def _get_api_key():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Run:\n"
            "    export GROQ_API_KEY='your-key-here'\n"
            "Get a free key at https://console.groq.com/keys"
        )
    return api_key


def call_llm(prompt: str, model: str, max_tokens: int = 500, retries: int = 3,
              reasoning_effort: str = None) -> str:
    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort

    time.sleep(4.0)

    last_error = None
    real_errors = 0
    max_rate_limit_retries = 15
    rate_limit_attempts = 0

    while real_errors < retries and rate_limit_attempts < max_rate_limit_retries:
        try:
            resp = requests.post(BASE_URL, headers=headers, json=body, timeout=30)
            if resp.status_code == 429:
                rate_limit_attempts += 1
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after else (2 ** min(real_errors, 5) * 5)
                print(f"[llm_client] rate limited, waiting {wait}s "
                      f"(rate-limit retry {rate_limit_attempts}/{max_rate_limit_retries})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            real_errors += 1
            wait = 2 ** real_errors
            print(f"[llm_client] attempt failed ({real_errors}/{retries} real errors): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    if rate_limit_attempts >= max_rate_limit_retries:
        raise RuntimeError(
            f"LLM call still rate-limited after {max_rate_limit_retries} rate-limit retries. "
            f"Check console.groq.com/settings/limits."
        )
    raise RuntimeError(f"LLM call failed after {real_errors} real errors: {last_error}")


def call_agent(prompt: str, max_tokens: int = 500) -> str:
    return call_llm(prompt, model=AGENT_MODEL, max_tokens=max_tokens, reasoning_effort="low")


def call_judge(prompt: str, max_tokens: int = 500) -> str:
    return call_llm(prompt, model=JUDGE_MODEL, max_tokens=max_tokens, reasoning_effort="low")