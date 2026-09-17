"""
prepare_judge_agreement_sample.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.pipeline import run_agent
from eval.llm_judge import judge_reply

golden = pd.read_csv("data/golden_set_labeled.csv", dtype=str)
sample = golden.sample(n=30, random_state=42)

print(f"Regenerating replies + judge scores for {len(sample)} rows (this will take a few minutes)...\n")

master = []
for i, row in enumerate(sample.itertuples()):
    print(f"[{i+1}/30] {row.tweet_id}...")
    result = run_agent(row.text)
    judge_scores = judge_reply(row.text, result["draft_reply"])
    master.append({
        "tweet_id": row.tweet_id,
        "message": row.text,
        "reply": result["draft_reply"],
        "judge_overall": judge_scores.get("overall"),
    })

Path("eval/judge_agreement_master.json").write_text(json.dumps(master, indent=2))

grading_df = pd.DataFrame(master)[["tweet_id", "message", "reply"]]
grading_df["your_score_1_to_5"] = ""
grading_df.to_csv("eval/judge_agreement_to_grade.csv", index=False)

print("\nDone. Saved eval/judge_agreement_to_grade.csv (grade this blind)")
print("and eval/judge_agreement_master.json (don't peek until you're done grading).")