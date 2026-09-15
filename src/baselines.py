"""
baselines.py

Two baselines the main pipeline must beat to justify its complexity:

1. TRIVIAL: always predicts the majority-class intent, always gives the same
   canned reply, always makes the same escalation decision. This is the
   floor -- if we don't beat this by a lot, something is broken.

2. SIMPLE: a real but basic non-LLM approach -- TF-IDF + Logistic Regression
   for intent classification, template-based reply (fill in a canned
   template based on predicted intent), and a keyword-rule escalation
   decision (no LLM at all). This tells us how much the LLM is actually
   buying us over cheap, fast, deterministic classical methods.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.escalation import SAFETY_NET_TERMS

CANNED_REPLY = "Thanks for reaching out. A member of our team will follow up with you shortly."

TEMPLATE_REPLIES = {
    "flight_delay_status": "We're sorry for the delay. Please check your flight status page for the latest updates.",
    "rebooking_connection": "We'll do our best to get you rebooked. Please see the Fly Delta app for available options.",
    "refund_compensation": "We understand, and we're happy to look into compensation options for you. Please DM us your confirmation number.",
    "checkin_seat_upgrade": "Please try checking in again via the Fly Delta app, or let us know if the issue persists.",
    "baggage_lost_item": "We're sorry about your bag. Please file a report through our baggage claim page and we'll follow up.",
    "service_complaint": "We're sorry to hear about your experience. We'd like to learn more -- please DM us the details.",
    "loyalty_miles_status": "You can check your SkyMiles balance and Medallion status details in the Fly Delta app.",
    "general_policy_info": "Thanks for your question -- please check delta.com or DM us and we'll help further.",
    "praise_or_noise": "Thank you so much, we really appreciate it!",
}


class TrivialBaseline:
    """Majority-class classifier, one canned reply, one fixed escalation decision."""

    def __init__(self, golden_set: pd.DataFrame):
        self.majority_intent = golden_set["your_intent_label"].mode()[0]

    def predict(self, message: str) -> dict:
        return {
            "intent": self.majority_intent,
            "draft_reply": CANNED_REPLY,
            "escalation_decision": "auto_handle",  # trivial baseline never escalates
            "escalation_reason": "Trivial baseline: never escalates.",
        }


class SimpleBaseline:
    """TF-IDF + Logistic Regression classifier, template reply, keyword-rule escalation."""

    def __init__(self, grounding_pool_texts, grounding_pool_labels=None):
        # NOTE: the simple baseline's classifier needs LABELED training data.
        # We don't have hand-labels for the full grounding pool (only the
        # golden set is labeled) -- so in practice this is trained on the
        # golden set itself via cross-validation in eval/run_eval.py, NOT
        # on the grounding pool. See run_eval.py for how this is handled
        # without leaking test folds into training folds.
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
        self.clf = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self._fitted = True

    def predict(self, message: str) -> dict:
        if not self._fitted:
            raise RuntimeError("SimpleBaseline must be fit() before predict()")
        X = self.vectorizer.transform([message])
        intent = self.clf.predict(X)[0]

        reply = TEMPLATE_REPLIES.get(intent, CANNED_REPLY)

        lowered = message.lower()
        escalate = any(term in lowered for term in SAFETY_NET_TERMS)
        decision = "escalate" if escalate else "auto_handle"
        reason = "Keyword-rule match" if escalate else "No red-flag keywords found"

        return {
            "intent": intent,
            "draft_reply": reply,
            "escalation_decision": decision,
            "escalation_reason": reason,
        }
