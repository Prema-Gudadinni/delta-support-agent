"""
escalation.py

Decides: auto_handle vs escalate_to_human, with a stated reason
(deliverable #3).

Design: HYBRID of rule-based safety net + LLM judgment.
- A small set of hard-coded red-flag terms (safety/medical/legal/discrimination)
  FORCE escalation no matter what -- these are too high-stakes to trust
  purely to an LLM's judgment call, and misses here are the most costly kind.
- Everything else is judged by the LLM using severity signals we identified
  from reading real examples (stranded, financial anger, repeated unresolved
  contact, explicit urgency language).

This mirrors a common real-world pattern: LLMs are good at nuanced judgment,
but for the highest-stakes categories you still want a deterministic
guardrail you can point to and audit.
"""

from src.llm_client import call_agent

# Deliberately conservative / high-recall list. False positives here (an
# unnecessary escalation) are cheap. False negatives (failing to escalate a
# real safety issue) are not. This asymmetry is WHY the list is a blunt
# keyword match rather than something fuzzier.
SAFETY_NET_TERMS = [
    "injur", "injury", "hurt", "bleeding", "medical", "emergency",
    "discriminat", "racist", "assault", "threat", "lawsuit", "sue", "legal action",
    "suicide", "unsafe", "danger",
]

ESCALATION_PROMPT_TEMPLATE = """You are deciding whether a Delta Air Lines customer support message should be
handled automatically by an AI agent, or escalated to a human agent.

Escalate if the message shows signs of: the customer being stranded/stuck with
nowhere to go, significant financial demands combined with anger or repeated
unresolved contact, explicit urgency ("emergency", time-critical), or a
complaint serious enough that a templated/automated reply would likely make
things worse.

Otherwise, auto-handle (routine questions, simple requests, praise, general info).

Customer message (intent classified as: {intent}):
"{message}"

Respond in exactly this format, two lines:
DECISION: auto_handle OR escalate
REASON: <one short sentence>
"""


def _check_safety_net(message: str) -> str | None:
    lowered = message.lower()
    for term in SAFETY_NET_TERMS:
        if term in lowered:
            return f"Safety-net keyword matched: '{term}'"
    return None


def decide_escalation(message: str, intent: str) -> dict:
    safety_reason = _check_safety_net(message)
    if safety_reason:
        return {"decision": "escalate", "reason": safety_reason, "source": "rule_safety_net"}

    prompt = ESCALATION_PROMPT_TEMPLATE.format(intent=intent, message=message)
    raw = call_agent(prompt, max_tokens=100).strip()

    decision = "escalate"  # fail-safe default if parsing fails: escalate, not auto-handle
    reason = "Could not parse LLM output cleanly; defaulted to escalate as a safe fallback."
    for line in raw.splitlines():
        if line.upper().startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().lower()
            decision = "auto_handle" if "auto" in val else "escalate"
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return {"decision": decision, "reason": reason, "source": "llm_judgment"}
