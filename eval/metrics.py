"""
metrics.py

Standard classification metrics for intent and escalation decisions.
Kept separate from LLM-judge (reply quality is subjective/generative,
these are hard-label comparisons).
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd


def classification_report(y_true, y_pred, labels=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "per_class": {
            label: {"precision": p, "recall": r, "f1": f, "support": int(s)}
            for label, p, r, f, s in zip(labels, *per_class)
        } if labels else None,
    }


def escalation_report(y_true, y_pred) -> dict:
    """
    Escalation is binary (auto_handle vs escalate), but NOT symmetric in
    cost: missing a true escalation (false negative) is much worse than
    an unnecessary escalation (false positive). So we report recall on
    the 'escalate' class specifically, not just overall accuracy --
    accuracy alone would hide a model that under-escalates.
    """
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=["auto_handle", "escalate"], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=["auto_handle", "escalate"])
    return {
        "accuracy": acc,
        "escalate_recall": recall[1],  # the critical number: did we catch true escalations?
        "escalate_precision": precision[1],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["auto_handle", "escalate"],
    }
