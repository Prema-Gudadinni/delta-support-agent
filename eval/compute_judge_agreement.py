"""
compute_judge_agreement.py (FIXED)
"""

import json
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

with open("eval/judge_agreement_master.json") as f:
    master = json.load(f)
master_lookup = {str(row["tweet_id"]): row["judge_overall"] for row in master}

graded = pd.read_csv("eval/judge_agreement_to_grade.csv", dtype=str)
graded["tweet_id"] = graded["tweet_id"].astype(str)

judge_scores = []
human_scores = []
matched_tweet_ids = []
skipped_null_judge = 0
skipped_ungraded = 0

for _, row in graded.iterrows():
    tid = row["tweet_id"]
    your_score = row.get("your_score_1_to_5")
    judge_score = master_lookup.get(tid)

    if pd.isna(your_score) or str(your_score).strip() == "":
        skipped_ungraded += 1
        continue
    if judge_score is None:
        skipped_null_judge += 1
        continue

    human_scores.append(int(float(your_score)))
    judge_scores.append(int(judge_score))
    matched_tweet_ids.append(tid)

print(f"Rows compared: {len(human_scores)} / {len(graded)}")
print(f"Skipped -- you left ungraded: {skipped_ungraded}")
print(f"Skipped -- judge score was null: {skipped_null_judge}")
print()

judge_arr = np.array(judge_scores)
human_arr = np.array(human_scores)

exact_match_rate = (judge_arr == human_arr).mean()
within_1_rate = (np.abs(judge_arr - human_arr) <= 1).mean()
correlation, p_value = spearmanr(judge_arr, human_arr)
mean_diff = (judge_arr - human_arr).mean()

print("=== JUDGE / HUMAN AGREEMENT ===")
print(f"Exact match rate:        {exact_match_rate:.1%}")
print(f"Within 1 point rate:     {within_1_rate:.1%}")
print(f"Spearman correlation:    {correlation:.3f} (p={p_value:.3f})")
print(f"Mean difference (judge - you): {mean_diff:+.2f}")

print("\nPer-row comparison (CORRECTED tweet_id alignment):")
for tid, j, h in zip(matched_tweet_ids, judge_scores, human_scores):
    diff = j - h
    flag = "  <-- BIG disagreement" if abs(diff) >= 2 else ""
    print(f"  {tid}: judge={j}, you={h}, diff={diff:+d}{flag}")