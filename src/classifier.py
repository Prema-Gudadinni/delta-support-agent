"""
classifier.py

Classifies a customer message into one of our 9 intents, AND (as of this
version) can also produce the escalation judgment in the SAME call.

WHY COMBINED: hitting Groq's free-tier 8,000 TPM cap made the eval run
impractically slow (~8-9 hours projected for 216 rows) when classify and
escalate were separate calls -- each call resent the full intent
definitions block, roughly doubling fixed overhead for no benefit. Merging
them into one call removes that duplication and cuts total agent-model
calls per row from 3 to 2 (combined classify+escalate, then draft).
Logged as a real decision -- a token-budget constraint changed the
architecture, not just the prompts.
"""

from src.llm_client import call_agent
from src.escalation import _check_safety_net

INTENTS = [
    "flight_delay_status",
    "rebooking_connection",
    "refund_compensation",
    "checkin_seat_upgrade",
    "baggage_lost_item",
    "service_complaint",
    "loyalty_miles_status",
    "general_policy_info",
    "praise_or_noise",
]

# Trimmed to short fragments -- this text is sent on EVERY classify call,
# so its length directly multiplies into token cost across all 216 rows.
INTENT_DEFINITIONS = """
flight_delay_status: delay/cancellation/mechanical status
rebooking_connection: needs alt flight, connection risk, availability
refund_compensation: wants money/voucher/credit back
checkin_seat_upgrade: check-in problems, seat, upgrade
baggage_lost_item: lost/delayed/damaged bag
service_complaint: staff/crew/aircraft complaint, not asking for refund
loyalty_miles_status: SkyMiles/MQD/Medallion questions
general_policy_info: general question, no active issue
praise_or_noise: compliment/thanks/off-topic, no action needed
""".strip()

CLASSIFY_PROMPT_TEMPLATE = """Classify this Delta Air Lines customer message into exactly one intent.

{definitions}

Message: "{message}"

Reply with ONLY the intent label, nothing else.
"""

# Combined prompt: gets intent AND escalation judgment in one call, saving
# a full duplicate "definitions" payload plus a second round-trip.
CLASSIFY_AND_ESCALATE_PROMPT_TEMPLATE = """Analyze this Delta Air Lines customer message.

Intents:
{definitions}

Escalate to a human if: stranded/no resolution path, financial demand + anger,
explicit urgency/emergency language, or repeated unresolved contact. Otherwise
auto_handle.

Message: "{message}"

Reply in EXACTLY this 3-line format, nothing else:
INTENT: <one label from the list above>
DECISION: <auto_handle or escalate>
REASON: <short phrase, only if escalate, else "n/a">
"""


def _match_intent(text: str) -> str:
    cleaned = text.strip().lower()
    for intent in INTENTS:
        if intent in cleaned:
            return intent
    return "unclassified"


def classify(message: str) -> str:
    """Standalone classify-only call. Kept for any code/tests that just need intent."""
    prompt = CLASSIFY_PROMPT_TEMPLATE.format(definitions=INTENT_DEFINITIONS, message=message)
    raw = call_agent(prompt, max_tokens=100).strip()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    last_line = lines[-1] if lines else raw
    result = _match_intent(last_line)
    if result == "unclassified":
        result = _match_intent(raw)
    if result == "unclassified":
        print(f"[classifier] UNPARSEABLE OUTPUT: {raw!r}")
    return result


def classify_and_judge_escalation(message: str) -> dict:
    """
    Combined call: returns {intent, escalation_decision, escalation_reason,
    escalation_source}. The rule-based safety net is still checked FIRST
    and for free (no tokens) -- if it fires, we skip the LLM's escalation
    judgment entirely and only ask it for intent (still one call, smaller
    prompt). This preserves our hybrid safety-net design from decision
    log #6 while cutting token cost.
    """
    safety_reason = _check_safety_net(message)

    prompt = CLASSIFY_AND_ESCALATE_PROMPT_TEMPLATE.format(
        definitions=INTENT_DEFINITIONS, message=message
    )
    raw = call_agent(prompt, max_tokens=150).strip()

    intent = "unclassified"
    decision = "escalate"  # fail-safe default
    reason = "Could not parse LLM output cleanly; defaulted to escalate as a safe fallback."

    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("INTENT:"):
            intent = _match_intent(line.split(":", 1)[1])
        elif line.upper().startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().lower()
            decision = "auto_handle" if "auto" in val else "escalate"
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    if intent == "unclassified":
        print(f"[classifier] UNPARSEABLE OUTPUT: {raw!r}")

    # Safety net overrides the LLM's escalation call regardless of what it said
    if safety_reason:
        decision = "escalate"
        reason = safety_reason
        source = "rule_safety_net"
    else:
        source = "llm_judgment"

    return {
        "intent": intent,
        "escalation_decision": decision,
        "escalation_reason": reason,
        "escalation_source": source,
    }
