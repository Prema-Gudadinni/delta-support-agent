"""
find_failures.py
"""

import json
import pandas as pd

with open("eval/results_final.json") as f:
    data = json.load(f)

golden = pd.read_csv("data/golden_set_labeled.csv", dtype=str)
golden["tweet_id"] = golden["tweet_id"].astype(str)
text_lookup = dict(zip(golden["tweet_id"], golden["text"]))
intent_label_lookup = dict(zip(golden["tweet_id"], golden["your_intent_label"].str.strip()))
escalate_label_lookup = dict(zip(
    golden["tweet_id"],
    golden["should_escalate (yes/no)"].apply(lambda v: "escalate" if str(v).strip().lower().startswith("y") else "auto_handle")
))

agent_rows = data["raw"]["agent"]

intent_mismatches = []
escalation_mismatches = []

for row in agent_rows:
    tid = str(row["tweet_id"])
    true_intent = intent_label_lookup.get(tid)
    true_escalate = escalate_label_lookup.get(tid)
    pred_intent = row["intent_pred"]
    pred_escalate = row["escalation_pred"]

    if true_intent and pred_intent != true_intent:
        intent_mismatches.append({
            "tweet_id": tid,
            "text": text_lookup.get(tid, ""),
            "your_label": true_intent,
            "agent_predicted": pred_intent,
        })

    if true_escalate and pred_escalate != true_escalate:
        escalation_mismatches.append({
            "tweet_id": tid,
            "text": text_lookup.get(tid, ""),
            "your_label": true_escalate,
            "agent_predicted": pred_escalate,
        })

print(f"=== INTENT MISMATCHES: {len(intent_mismatches)} / {len(agent_rows)} ===\n")
for m in intent_mismatches:
    print(f"[{m['tweet_id']}] your_label={m['your_label']} | agent={m['agent_predicted']}")
    print(f"  \"{m['text'][:150]}\"\n")

print(f"\n=== ESCALATION MISMATCHES: {len(escalation_mismatches)} / {len(agent_rows)} ===\n")
for m in escalation_mismatches:
    print(f"[{m['tweet_id']}] your_label={m['your_label']} | agent={m['agent_predicted']}")
    print(f"  \"{m['text'][:150]}\"\n")

pd.DataFrame(intent_mismatches).to_csv("eval/intent_mismatches.csv", index=False)
pd.DataFrame(escalation_mismatches).to_csv("eval/escalation_mismatches.csv", index=False)
print("Saved eval/intent_mismatches.csv and eval/escalation_mismatches.csv")