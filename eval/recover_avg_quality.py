"""
recover_avg_quality.py
"""

import json

with open("eval/results_final.json") as f:
    data = json.load(f)

agent_rows = data["raw"]["agent"]

scores = [r["judge_scores"].get("overall") for r in agent_rows if r["judge_scores"].get("overall") is not None]
missing = len(agent_rows) - len(scores)

print(f"Valid judge scores: {len(scores)} / {len(agent_rows)}")
print(f"Missing/unparseable judge scores: {missing}")
print(f"Avg reply quality (judge, 1-5): {sum(scores)/len(scores):.2f}")