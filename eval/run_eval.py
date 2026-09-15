"""
run_eval.py

The main script that produces your headline results. Run this after:
1. You've finished hand-labeling data/golden_set_to_label.csv
2. Saved it as data/golden_set_labeled.csv
3. Set GROQ_API_KEY

Usage:
    python3 -m eval.run_eval

Produces:
    eval/checkpoint.jsonl -- one line per completed row, written incrementally
    eval/results.json -- full results for agent + both baselines (final summary)
    Printed summary table to stdout

CHECKPOINTING: results are saved to eval/checkpoint.jsonl as each row
completes, not just at the end. If the script crashes or hits a rate-limit
wall partway through (a real risk on Groq's free tier -- see decision log),
just rerun it: already-completed rows are skipped automatically. This
turned "lost all progress at row 150, start over" into "rerun later,
picks up at row 151" -- worth doing given how tight the free-tier daily
request cap is relative to ~850+ total calls needed for a full run.
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_golden_set
from src.pipeline import run_agent
from src.baselines import TrivialBaseline, SimpleBaseline
from eval.metrics import classification_report, escalation_report
from eval.llm_judge import judge_reply
from src.classifier import INTENTS

CHECKPOINT_PATH = Path("eval/checkpoint.jsonl")


def normalize_escalation_label(val: str) -> str:
    v = str(val).strip().lower()
    return "escalate" if v.startswith("y") else "auto_handle"


def load_checkpoint() -> dict:
    """Returns {tweet_id: row_result} for already-completed rows."""
    if not CHECKPOINT_PATH.exists():
        return {}
    done = {}
    with open(CHECKPOINT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[row["tweet_id"]] = row
    return done


def append_checkpoint(row_result: dict):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(json.dumps(row_result) + "\n")


def main():
    golden = load_golden_set()
    print(f"Loaded golden set: {len(golden)} labeled examples")

    sample_size = os.environ.get("EVAL_SAMPLE_SIZE")
    if sample_size:
        golden = golden.sample(n=min(int(sample_size), len(golden)), random_state=1)
        print(f"DEBUG MODE: running on a random subsample of {len(golden)} rows "
              f"(set via EVAL_SAMPLE_SIZE env var). Unset it for the full run.")

    y_true_intent = golden["your_intent_label"].str.strip().tolist()
    y_true_escalation = golden["should_escalate (yes/no)"].apply(normalize_escalation_label).tolist()

    trivial = TrivialBaseline(golden)
    simple = SimpleBaseline(golden["text"].tolist())
    simple.fit(golden["text"].tolist(), y_true_intent)

    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"Resuming from checkpoint: {len(checkpoint)} rows already completed, skipping those.")

    results = {"agent": [], "trivial_baseline": [], "simple_baseline": []}
    remaining = 0
    for _, row in golden.iterrows():
        if row["tweet_id"] in checkpoint:
            remaining += 0
        else:
            remaining += 1
    print(f"{remaining} rows still to process this run.\n")

    for i, (_, row) in enumerate(golden.iterrows()):
        tweet_id = row["tweet_id"]
        message = row["text"]

        if tweet_id in checkpoint:
            cp = checkpoint[tweet_id]
            results["agent"].append(cp["agent"])
            results["trivial_baseline"].append(cp["trivial_baseline"])
            results["simple_baseline"].append(cp["simple_baseline"])
            continue

        print(f"[{i+1}/{len(golden)}] Processing tweet_id {tweet_id}...")

        agent_out = run_agent(message)
        judge_scores = judge_reply(message, agent_out["draft_reply"])
        agent_result = {
            "tweet_id": tweet_id,
            "intent_pred": agent_out["intent"],
            "escalation_pred": agent_out["escalation_decision"],
            "judge_scores": judge_scores,
        }

        triv_out = trivial.predict(message)
        trivial_result = {
            "tweet_id": tweet_id,
            "intent_pred": triv_out["intent"],
            "escalation_pred": triv_out["escalation_decision"],
        }

        simp_out = simple.predict(message)
        simple_result = {
            "tweet_id": tweet_id,
            "intent_pred": simp_out["intent"],
            "escalation_pred": simp_out["escalation_decision"],
        }

        results["agent"].append(agent_result)
        results["trivial_baseline"].append(trivial_result)
        results["simple_baseline"].append(simple_result)

        # Save progress immediately -- this is the whole point of checkpointing
        append_checkpoint({
            "tweet_id": tweet_id,
            "agent": agent_result,
            "trivial_baseline": trivial_result,
            "simple_baseline": simple_result,
        })

    # --- Compute metrics per system
    summary = {}
    for system_name in ["agent", "trivial_baseline", "simple_baseline"]:
        preds_intent = [r["intent_pred"] for r in results[system_name]]
        preds_escalation = [r["escalation_pred"] for r in results[system_name]]

        summary[system_name] = {
            "classification": classification_report(y_true_intent, preds_intent, labels=INTENTS),
            "escalation": escalation_report(y_true_escalation, preds_escalation),
        }

    if results["agent"] and results["agent"][0]["judge_scores"].get("overall") is not None:
        overall_scores = [r["judge_scores"]["overall"] for r in results["agent"] if r["judge_scores"].get("overall")]
        summary["agent"]["avg_reply_quality"] = sum(overall_scores) / len(overall_scores)

    Path("eval/results.json").write_text(json.dumps({"summary": summary, "raw": results}, indent=2))

    print("\n=== SUMMARY ===")
    for system_name, s in summary.items():
        print(f"\n{system_name}:")
        print(f"  Intent accuracy: {s['classification']['accuracy']:.3f}")
        print(f"  Intent macro-F1: {s['classification']['macro_f1']:.3f}")
        print(f"  Escalation accuracy: {s['escalation']['accuracy']:.3f}")
        print(f"  Escalation recall (catching true escalations): {s['escalation']['escalate_recall']:.3f}")
        if "avg_reply_quality" in s:
            print(f"  Avg reply quality (judge, 1-5): {s['avg_reply_quality']:.2f}")

    print("\nFull results saved to eval/results.json")
    print("NOTE: once this completes successfully, you can delete eval/checkpoint.jsonl")


if __name__ == "__main__":
    main()
