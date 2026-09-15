"""
llm_judge.py

Grades draft reply QUALITY (not intent correctness -- that's a hard label,
handled in metrics.py). Reply quality is subjective/generative, so we use
an LLM judge against a rubric, then separately measure how much the judge
agrees with a small human-graded sample (deliverable requirement:
"evidence of how well your judge agrees with a human").

IMPORTANT: uses JUDGE_MODEL from llm_client.py, which is deliberately a
DIFFERENT model than AGENT_MODEL (the one drafting replies). Grading your
own homework with the same model risks self-preference bias -- inflating
your headline reply-quality score. This is one of the things we
specifically designed against; it should show up as a defensible point in
your report, not just a paragraph of good intentions.
"""

from src.llm_client import call_judge

JUDGE_PROMPT_TEMPLATE = """Grade this Delta customer support reply, 1-5 each dimension.

Message: "{message}"
Reply: "{reply}"

RELEVANCE (addresses the message?), GROUNDEDNESS (no invented facts?), TONE (professional/empathetic?), ACTIONABILITY (clear next step?).

Format exactly:
RELEVANCE: <1-5>
GROUNDEDNESS: <1-5>
TONE: <1-5>
ACTIONABILITY: <1-5>
OVERALL: <1-5>
RATIONALE: <one sentence>
"""


def judge_reply(message: str, reply: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(message=message, reply=reply)
    raw = call_judge(prompt, max_tokens=120).strip()

    scores = {}
    rationale = ""
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().upper()
        val = val.strip()
        if key == "RATIONALE":
            rationale = val
        elif key in ("RELEVANCE", "GROUNDEDNESS", "TONE", "ACTIONABILITY", "OVERALL"):
            try:
                scores[key.lower()] = int(val[0])  # first digit, defensive parse
            except (ValueError, IndexError):
                scores[key.lower()] = None

    scores["rationale"] = rationale
    return scores


def judge_agreement_with_human(judge_scores: list, human_scores: list) -> dict:
    """
    Measures agreement between LLM judge's OVERALL score and a human's
    OVERALL score on the SAME sample of replies.

    You (the human) must grade a subset yourself -- e.g. 30 of the 216
    golden-set replies -- on the same 1-5 OVERALL scale, independently,
    BEFORE looking at the judge's scores. This is what the deliverable
    means by "evidence of how well your judge agrees with a human" --
    it can't be skipped or faked with the judge grading itself twice.
    """
    import numpy as np
    from scipy.stats import spearmanr

    judge_arr = np.array(judge_scores)
    human_arr = np.array(human_scores)

    exact_match_rate = (judge_arr == human_arr).mean()
    within_1_rate = (np.abs(judge_arr - human_arr) <= 1).mean()
    correlation, _ = spearmanr(judge_arr, human_arr)

    return {
        "n": len(judge_scores),
        "exact_match_rate": float(exact_match_rate),
        "within_1_point_rate": float(within_1_rate),
        "spearman_correlation": float(correlation),
    }
