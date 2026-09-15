# Delta AI Support Agent -- Report

## 1. Problem framing

**What "good" means for this brand:**
[Write 2-3 sentences: for Delta specifically, what does a good support
interaction look like? E.g. correctly triaging urgency, grounding replies in
real policy/tone rather than generic corporate-speak, catching the small
percentage of messages that are actually crises.]

**What I chose not to build:**
- Multi-turn conversation handling (only first customer message -> Delta
  reply pairs; see decision log #2 on scope).
- Embedding-based retrieval (used TF-IDF instead; decision log #8).
- [Add anything else you consciously scoped out]

## 2. Taxonomy derivation

[Briefly describe: read 65 real messages across two samples, iterated the
taxonomy as new patterns emerged (loyalty questions, off-topic noise).
Link to or summarize the two sample reviews.]

Final taxonomy: flight_delay_status, rebooking_connection, refund_compensation,
checkin_seat_upgrade, baggage_lost_item, service_complaint, loyalty_miles_status,
general_policy_info, praise_or_noise.

## 3. Results vs. baselines

[Run `python3 -m eval.run_eval` and paste the summary table here once your
golden set is labeled. Include: intent accuracy/F1, escalation accuracy +
escalate-recall specifically, avg reply quality, for all three systems:
agent, trivial baseline, simple baseline.]

| System | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalate Recall | Avg Reply Quality |
|---|---|---|---|---|---|
| Agent | | | | | |
| Trivial baseline | | | | | |
| Simple baseline | | | | | |

## 4. Failure analysis -- top 5 failure modes

[After running eval, look at eval/results.json for the rows where agent
predictions != your labels, or where judge scores were low. For each of
5 real failure modes, give:
- A short name
- A real example (tweet_id + text)
- Your hypothesis for WHY it failed]

1. **[Failure mode name]**
   - Example: [tweet_id] "..."
   - Hypothesis: ...

2. ...

## 5. What is misleading about my headline number? (MANDATORY)

[This is the section that matters most. CONFIRMED, not hypothetical -- start here:

- **Same model drafts AND grades replies (gemini-2.0-flash-lite for both).**
  No self-preference-bias mitigation is in effect. The reply-quality score
  in your results table is likely inflated relative to what an independent
  judge would give -- quantify this if you can (e.g., spot-check a few
  low/borderline replies and see if you personally agree with the score
  the model gave itself). See decision log #9c for why this happened.

Other angles worth checking against your real results:

- Golden set is stratified, not randomly sampled -- headline accuracy may
  not reflect true intent frequency in production (rare intents are
  over-represented relative to reality).
- SimpleBaseline is fit AND evaluated on the same 216 rows (no
  cross-validation) -- its number is optimistic, making the LLM agent's
  relative improvement look smaller (or larger) than it really is.
- LLM judge grading reply quality was only checked against a human on a
  SUBSET -- if judge/human agreement is weak on that subset, the full
  reply-quality number is not very trustworthy.
- praise_or_noise messages are the easiest to classify correctly (nearly
  keyword-matchable) -- if they're ~20% of the golden set, overall accuracy
  is partly inflated by an easy class that doesn't reflect real support
  difficulty.
- Escalation recall on a SMALL golden set: if only a handful of true
  "escalate" examples exist in 216 rows, the recall number has high
  variance -- one or two misses swings it a lot.

Write this in your own voice, backed by actual numbers from your run --
don't just copy the bullets above without checking which are real.]

## 6. What I'd do next with one more week

[e.g. multi-turn thread handling, embedding-based retrieval, cross-validated
baseline, larger/re-sampled golden set, active learning on failure cases]
