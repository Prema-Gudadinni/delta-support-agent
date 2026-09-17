"""
pipeline.py

Orchestrates the full agent. Classify + escalate + draft are ALL merged
into a single agent-model call (previously 2 separate calls) to cut
token overhead and pacing delay under Groq's free-tier TPM cap.
"""

from src.llm_client import call_agent
from src.escalation import _check_safety_net
from src.reply_drafter import retrieve_similar
from src.classifier import INTENTS, INTENT_DEFINITIONS, _match_intent

COMBINED_PROMPT_TEMPLATE = """You are Delta Air Lines' AI support agent. Do 3 things for this customer message.

Intents:
{definitions}

Escalate if: stranded/no resolution path, financial demand + anger, explicit urgency, or repeated unresolved contact. Otherwise auto_handle.

Past examples of how Delta replied to similar messages:
{examples}

Customer message: "{message}"

Reply in EXACTLY this format, nothing else:
INTENT: <one label>
DECISION: <auto_handle or escalate>
REASON: <short phrase if escalate, else "n/a">
REPLY: <your drafted reply, same tone/style as examples, concise, don't invent facts>
"""


def run_agent(message: str) -> dict:
    safety_reason = _check_safety_net(message)

    similar = retrieve_similar(message, k=2)
    examples_text = "\n\n".join(
        f'Customer: "{s["similar_customer_text"][:150]}"\nDelta: "{s["delta_reply"][:150]}"'
        for s in similar
    )

    prompt = COMBINED_PROMPT_TEMPLATE.format(
        definitions=INTENT_DEFINITIONS, examples=examples_text, message=message
    )
    raw = call_agent(prompt, max_tokens=250).strip()

    intent = "unclassified"
    decision = "escalate"
    reason = "Could not parse LLM output cleanly; defaulted to escalate as a safe fallback."
    reply = ""

    lines = raw.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.upper().startswith("INTENT:"):
            intent = _match_intent(stripped.split(":", 1)[1])
        elif stripped.upper().startswith("DECISION:"):
            val = stripped.split(":", 1)[1].strip().lower()
            decision = "auto_handle" if "auto" in val else "escalate"
        elif stripped.upper().startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()
        elif stripped.upper().startswith("REPLY:"):
            reply = stripped.split(":", 1)[1].strip()
            if i + 1 < len(lines):
                reply = (reply + " " + " ".join(l.strip() for l in lines[i+1:])).strip()
            break

    if intent == "unclassified" or not reply:
        print(f"[pipeline] PARSING ISSUE on output: {raw!r}")

    if safety_reason:
        decision = "escalate"
        reason = safety_reason
        source = "rule_safety_net"
    else:
        source = "llm_judgment"

    return {
        "message": message,
        "intent": intent,
        "draft_reply": reply,
        "grounding_examples": similar,
        "escalation_decision": decision,
        "escalation_reason": reason,
        "escalation_source": source,
    }


if __name__ == "__main__":
    test_message = "My flight DL2703 was changed last night and I still haven't heard back. Will I make my connection?"
    result = run_agent(test_message)
    import json
    print(json.dumps(result, indent=2))