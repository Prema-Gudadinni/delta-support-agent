# Delta AI Support Agent

An AI agent that classifies customer support messages sent to @Delta on Twitter,
drafts a reply grounded in how Delta has historically resolved similar issues,
and decides whether to auto-handle or escalate to a human.

## How the system works

```
customer message
      |
      v
[classifier]  -> intent (1 of 9 categories, derived from reading real data)
      |
      v
[reply_drafter] -> retrieves similar past (customer, Delta-reply) pairs via
      |             TF-IDF similarity, uses them as grounding, drafts a new reply
      v
[escalation]  -> rule-based safety net (injury/legal/discrimination keywords)
      |           + LLM judgment for everything else -> auto_handle / escalate + reason
      v
   final output
```

## Why these design choices (short version -- full reasoning in report.md)

- **Intents were derived from data, not assumed.** See report.md for the two
  manual reading passes (65 real messages) that produced the 9-category taxonomy.
- **Leakage-safe grounding.** The 216 golden-set examples are excluded from the
  pool the reply drafter can retrieve from (`src/data_loader.get_grounding_pool`).
  Verified programmatically (overlap check = 0).
- **Agent model != judge model.** Different model families on Groq
  (llama-3.1-8b-instant vs llama-3.3-70b-versatile) grade vs. draft, to
  avoid self-preference bias inflating reply-quality scores. See
  `src/llm_client.py`.
- **Escalation is a hybrid**, not pure LLM judgment: deterministic keyword
  safety net for the highest-stakes categories (injury, legal, discrimination),
  LLM judgment for everything else.

## Setup (should take ~5 minutes)

Uses Groq's API (free tier, no credit card or phone verification required):

1. Get a free key at https://console.groq.com/keys
2. ```bash
   pip install -r requirements.txt
   export GROQ_API_KEY="your-key-here"
   ```

Note: free-tier rate limits apply (~30 requests/min, ~14,400/day on Llama
models). `src/llm_client.py` already retries with backoff on 429s, but
running the full 216-row golden set will take some minutes due to pacing --
this is expected, not a bug.

## Data

- `data/delta_threads.csv` -- filtered subset of the Kaggle "Customer Support
  on Twitter" dataset, containing only Delta's threads (83,786 rows).
- `data/golden_set_to_label.csv` -- 216 stratified-sampled customer messages
  for hand-labeling.
- `data/golden_set_labeled.csv` -- **you must produce this** by hand-labeling
  the file above (see "Hand-labeling" section below), then saving with this
  exact filename.

## Reproducing headline results (should take <15 minutes on a subsample)

```bash
python3 -m eval.run_eval
```

This runs the agent + both baselines against your labeled golden set and
prints a summary table, plus saves full results to `eval/results.json`.

## Hand-labeling the golden set

Open `data/golden_set_to_label.csv`. For each row, fill in:
- `your_intent_label`: one of the 9 intents (see `src/classifier.py` INTENTS)
- `should_escalate (yes/no)`
- `escalation_reason`
- `notes`: flag anything ambiguous, multi-intent, or off-topic -- this feeds
  directly into the failure analysis section of the report.

Then save as `data/golden_set_labeled.csv`.

## Project structure

```
src/
  data_loader.py    -- loads data, builds leakage-safe train/eval split
  llm_client.py      -- single point of contact with the LLM API
  classifier.py      -- intent classification
  reply_drafter.py   -- retrieval-grounded reply generation
  escalation.py      -- auto-handle vs escalate decision
  pipeline.py         -- ties the three together
  baselines.py         -- trivial + simple (non-LLM) baselines
eval/
  metrics.py          -- classification + escalation metrics
  llm_judge.py         -- LLM-as-judge for reply quality + human agreement calc
  run_eval.py          -- main evaluation script
report/                -- report.md, decision_log.md (see deliverables)
```

## Known limitations (see report.md for the full "what's misleading" analysis)

- Only captures first-turn (customer -> Delta) pairs, not full multi-turn threads.
- SimpleBaseline is fit and evaluated on the same 216 golden-set rows
  (no cross-validation) -- likely overstates its real performance.
- Reply retrieval uses TF-IDF, not embeddings -- misses semantic matches that
  share no vocabulary with the query.
