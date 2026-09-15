"""
llm_client.py

Single point of contact with the LLM API. Every other module (classifier,
reply_drafter, escalation, llm_judge) calls through here instead of hitting
the API directly.

WHY: keeps model choice, retry logic, and error handling in one place.
Also makes it trivial to enforce our "agent model != judge model" rule
(see eval/llm_judge.py) -- you set them independently here and the rest
of the codebase never has to think about it.

Uses Groq's API (free tier, no credit card / phone verification required).
Groq is OpenAI-compatible (standard Bearer auth, /chat/completions shape),
which is why this implementation is simpler than the Gemini version we
started with -- that one hit real auth/model-availability friction that
cost significant debugging time (see decision log #9/#9b/#9c). Switched
providers rather than keep debugging, given the assignment deadline.

Requires a GROQ_API_KEY environment variable to actually run.
Get a free key: https://console.groq.com/keys (no credit card needed).
"""

import os
import time
import requests

# Two separate model settings -- deliberately kept independent, and this
# time genuinely different model FAMILIES (not just tiers of the same
# model), giving real self-preference-bias mitigation.
AGENT_MODEL = "openai/gpt-oss-20b"       # fast, replaces deprecated llama-3.1-8b-instant
JUDGE_MODEL = "qwen/qwen3.6-27b"          # different model family entirely (not just different size within
                                           # the same family) -- stronger self-preference-bias mitigation

BASE_URL = "https://api.groq.com/openai/v1/chat/completions"


def _get_api_key():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Run:\n"
            "    export GROQ_API_KEY='your-key-here'\n"
            "before running any pipeline or eval script.\n"
            "Get a free key at https://console.groq.com/keys"
        )
    return api_key


def call_llm(prompt: str, model: str, max_tokens: int = 500, retries: int = 3,
              reasoning_effort: str = None) -> str:
    """
    Single LLM call with basic retry on transient failures.

    reasoning_effort: for Groq's reasoning models (gpt-oss-*, qwen3.6-27b),
    reasoning tokens are billed against the SAME output budget as the final
    answer -- so a "small" max_tokens can get entirely consumed by internal
    reasoning, leaving an EMPTY final answer. This isn't a max_tokens
    problem, it's a reasoning-budget problem -- fixed by explicitly
    dialing down reasoning_effort rather than just raising max_tokens.
    """
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

    time.sleep(4.0)  # proactive pacing -- tuned for the 8,000 TPM cap (the tighter
                     # constraint vs. the 30 RPM cap, given our prompts average
                     # several hundred tokens each with reasoning included)  # proactive pacing -- avoids most 429s instead of just reacting to them

    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.post(BASE_URL, headers=headers, json=body, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after else (2 ** attempt * 5)
                print(f"[llm_client] rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            print(f"[llm_client] attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_error}")


def call_agent(prompt: str, max_tokens: int = 500) -> str:
    # low reasoning effort: classify/draft/escalate are simple enough tasks
    # that heavy internal reasoning just burns budget without improving
    # quality -- and was actively breaking output by consuming the whole
    # token budget before any answer text appeared.
    return call_llm(prompt, model=AGENT_MODEL, max_tokens=max_tokens, reasoning_effort="low")


def call_judge(prompt: str, max_tokens: int = 500) -> str:
    # qwen3 models support "none" to fully disable thinking mode -- avoids
    # both the token-budget problem AND qwen's tendency to leak <think>...
    # </think> tags directly into the content field for this task.
    return call_llm(prompt, model=JUDGE_MODEL, max_tokens=max_tokens, reasoning_effort="none")
