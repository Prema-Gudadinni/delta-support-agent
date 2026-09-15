"""
pipeline.py

Orchestrates the full agent: classify+escalate (combined call) -> draft reply.

NOTE: classify and escalate were originally two separate calls; merged into
one (classify_and_judge_escalation) to cut token usage under Groq's free-tier
TPM cap -- see decision log and src/classifier.py docstring.
"""

from src.classifier import classify_and_judge_escalation
from src.reply_drafter import draft_reply


def run_agent(message: str) -> dict:
    classification = classify_and_judge_escalation(message)
    draft = draft_reply(message)

    return {
        "message": message,
        "intent": classification["intent"],
        "draft_reply": draft["reply"],
        "grounding_examples": draft["grounding_examples"],
        "escalation_decision": classification["escalation_decision"],
        "escalation_reason": classification["escalation_reason"],
        "escalation_source": classification["escalation_source"],
    }


if __name__ == "__main__":
    test_message = "My flight DL2703 was changed last night and I still haven't heard back. Will I make my connection?"
    result = run_agent(test_message)
    import json
    print(json.dumps(result, indent=2))
