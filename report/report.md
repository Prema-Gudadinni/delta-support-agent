# Delta AI Support Agent -- Report

## 1. Problem framing

**What "good" means for this brand:**
For Delta specifically, "good" means three things working together: correctly
identifying what a customer actually needs (not just keyword-matching their
complaint), drafting a reply that sounds like Delta's real voice -- grounded
in how the airline has actually resolved similar issues before, not generic
corporate boilerplate -- and, most importantly, correctly triaging the small
fraction of messages that represent a genuine crisis (stranded passengers,
safety concerns, repeated unresolved contact) versus the majority that are
routine and can be handled with a template. Getting the routine cases right
is necessary but not sufficient; missing a real escalation is the costliest
possible failure for a support system like this, which is why escalation
recall -- not just overall accuracy -- was treated as the headline metric
throughout this project.

**What I chose not to build:**
- Multi-turn conversation handling (only first customer message -> Delta
  reply pairs; see decision log on scope).
- Embedding-based retrieval (used TF-IDF instead; see decision log and
  Section 4, failure mode #1, for the real cost of this choice).
- A fully independent judge model/provider (ended up on the same model
  family for agent and judge after repeated free-tier provider issues;
  see decision log and Section 5).
- Cross-validated baselines (SimpleBaseline fit and evaluated on the same
  216 rows; a known, disclosed limitation rather than a fair comparison).

## 2. Taxonomy derivation

The 9-intent taxonomy was not assumed from a generic "airline support"
template -- it was built by reading two random samples of real Delta
customer messages from the dataset (35 messages, then a second batch of 30
messages, 65 total). The first pass surfaced the obvious categories:
flight delays, rebooking, refunds, check-in/seat issues, baggage, and
general service complaints. The second pass forced two revisions: adding
`loyalty_miles_status` as its own category once it became clear that
SkyMiles/MQD/Medallion questions were common enough and distinct enough
from generic policy questions to deserve their own bucket, and adding
`praise_or_noise` after noticing that roughly 20% of real messages were
compliments, thank-yous, or off-topic mentions -- not actionable support
requests at all. Forcing these into "real" intent categories would have
artificially inflated classifier accuracy on an easy class while hiding
that the model wasn't being meaningfully tested. This decision is directly
validated by Section 4's failure analysis: praise_or_noise messages turned
out to have their own distinct failure mode (judge rubric mismatch) and
confusion pattern (sarcasm misread as genuine praise) that would have been
invisible if they'd been lumped into other categories.

Final taxonomy: flight_delay_status, rebooking_connection, refund_compensation,
checkin_seat_upgrade, baggage_lost_item, service_complaint, loyalty_miles_status,
general_policy_info, praise_or_noise.

## 3. Results vs. baselines

Full 216-row golden set, single consistent pipeline version throughout
(see decision log for the earlier debugging saga that led to this final
architecture).

| System | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Escalate Recall | Avg Reply Quality |
|---|---|---|---|---|---|
| Agent | 78.7% | 78.3% | 84.3% | **27.8%** | 4.04 (on 176/216 rows with valid judge scores; see below) |
| Trivial baseline | 20.4% | 3.8% | 83.3% | 0.0% | -- |
| Simple baseline (TF-IDF+LogReg) | **94.9%** | **95.0%** | 82.9% | 11.1% | -- |

Two results immediately stand out and are discussed in Section 5: the
simple non-LLM baseline outperforms the LLM agent on intent classification,
and the agent's escalation recall is low despite reasonably high overall
escalation accuracy.

## 4. Failure analysis -- top 5 failure modes

**1. Lexical-overlap retrieval surfaces topically irrelevant grounding.**
The reply-drafter uses TF-IDF cosine similarity to find past examples to
ground new replies in. This works well when messages share meaningful
vocabulary, but fails when they share only generic phrasing. Example: a
customer asking "will I make my connection?" (a rebooking question)
retrieved grounding examples about *lost items*, purely because both
happened to contain the phrase "haven't heard anything." The retrieved
examples were confidently used as tone/style reference despite being
topically unrelated.
- Hypothesis: TF-IDF captures word overlap, not semantic meaning.
  Embedding-based retrieval (see Section 6) would likely fix this.

**2. The LLM judge structurally under-scores praise/thank-you replies.**
From the judge-agreement check (30-row human-graded sample): tweet_id
611555 ("@Delta Thank You!" -> "Thank you for reaching out! Have a
great day.") was scored 5/5 by me, but only 3/5 by the judge -- the single
largest disagreement in the sample.
- Hypothesis: the judge rubric scores four fixed dimensions including
  "Actionability" (does the reply give a clear next step?) for every
  message uniformly. A thank-you message doesn't need a next step, but
  the rubric likely penalizes the reply anyway for lacking one. This
  could be systematically deflating quality scores specifically for
  praise_or_noise messages -- worth checking against the full 216-row
  set, not just this sample.

**3. Sarcasm and irony are misread as genuine positive sentiment.**
Two clear examples: tweet_id 1711516 ("Thanks @Delta - the on again/off
again status... allowed me a lovely run to the gate....") is sarcastic
frustration about a chaotic delay, labeled flight_delay_status by me but
classified praise_or_noise by the agent. Tweet_id 1249531 ("@Delta thanks
for a terrible experience") is unambiguous sarcasm, same misclassification.
- Hypothesis: the classifier appears to weight the surface presence of
  "thanks"/positive words heavily, without parsing tone or contradiction
  with the rest of the message. This is a real limitation even for an
  LLM-based classifier, not just a keyword system.

**4. Systematic boundary confusion between loyalty_miles_status and
general_policy_info.** Not random noise -- recurs specifically on this
pair across 4+ examples: tweet_id 1688754 (Flying Blue miles purchase
question), 714278 (how to contact the Medallion desk), 207195 and 909836
(both about SkyMiles Marketplace status, both mismatched the same way).
- Hypothesis: these questions are genuinely ambiguous under my own
  taxonomy -- they involve the loyalty program but ask "how do I do X"
  rather than "what is my status/balance," blurring the intended
  boundary. This is as much a taxonomy design limitation as a model
  failure, and worth naming honestly.

**5. Under-escalation on inferred/compound severity, vs. explicit
keywords -- the primary driver of the 27.8% escalation recall.**
The rule-based safety net catches explicit danger language reliably. But
12+ under-escalated cases share a different pattern: severity that
requires COMBINING facts rather than matching a keyword. Examples:
tweet_id 796635 (stuck on tarmac + "will likely miss connection" --
requires connecting "tarmac delay" to "connection risk"), tweet_id 764245
("6 weeks... 10+ calls... 40 minutes... no news" -- requires recognizing
a pattern of repeated failure across a whole message, not a single
phrase), tweet_id 1858265 ("a month ago... need 3 more weeks" -- same
compounding-duration pattern).
- Hypothesis: the LLM's escalation judgment is weaker at inferring
  severity that isn't stated in an obvious trigger phrase. This is the
  single most consequential failure mode found, since under-escalation
  is the costlier error type by design.

## 5. What is misleading about my headline number?

**The simple TF-IDF+LogisticRegression baseline (94.9% intent accuracy)
substantially outperforms the LLM agent (78.7%) -- but this comparison is
not fully fair, in a way that cuts in BOTH directions.** SimpleBaseline was
fit AND evaluated on the same 216-row golden set with no train/test split
or cross-validation (a known limitation logged from the start) -- its
94.9% is likely optimistic, since it was tested on data it effectively
memorized patterns from. A fair comparison would need k-fold cross-
validation on the baseline. That said, even a generously-discounted
baseline number is unlikely to fully close a 16-point gap, so the
underlying finding -- that a cheap, fast, interpretable classical method
is highly competitive with an LLM on this specific 9-class task -- likely
still holds and deserves to be taken seriously, not dismissed.

**Escalation recall (27.8%) is the most operationally important number in
this whole report, and it is currently unreliable in a subtle way: my
golden set was stratified-sampled by keyword-heuristic bucket, not
randomly sampled from the true distribution.** This means the *proportion*
of true escalation cases in my 216-row set does not necessarily match
production reality -- it could be over- or under-representing
escalation-worthy messages relative to what Delta actually sees. The
27.8% recall is a reasonably reliable READ on the model's current
weakness (confirmed by consistent patterns across many examples in
Section 4), but the raw escalation rate/precision numbers should not be
read as "X% of real customer messages need escalation."

**Self-preference bias in the judge was only partially controlled for.**
Agent (gpt-oss-20b) and judge (gpt-oss-120b) are different model sizes but
the SAME model family/provider, not a fully independent judge (a
deadline-driven pivot after repeated free-tier provider issues -- see
decision log). The judge-agreement check (56% exact match, 96% within
one point, Spearman correlation 0.48) suggests reasonable but not perfect
alignment with my own judgment, and the one clear disagreement found
(tweet_id 611555) points to a structural rubric issue (Section 4, finding
#2) rather than random noise -- meaning the reply-quality average may be
systematically, not just randomly, off for certain intent categories.

**A meaningful fraction of judge gradings failed to parse into a numeric
score entirely: 40 out of 216 rows (18.5%).** These rows are silently
excluded from the average reply-quality score of 4.04, meaning that
number is computed on 176/216 rows, not genuinely all of them. This is
large enough to matter, not a rounding-error edge case -- if these 40
failures aren't randomly distributed (e.g. if they correlate with longer
or more complex draft replies, which are more likely to produce malformed
judge output), the true average reply quality across all 216 rows could
be meaningfully different from 4.04 in either direction. A stricter
project would investigate whether these 40 failures cluster by intent
category or reply length before trusting the headline 4.04 number at all.

## 6. What I'd do next with one more week

- **Switch retrieval from TF-IDF to embeddings** (e.g. a small sentence-
  embedding model) to fix the topical-mismatch grounding failures found
  in Section 4 -- likely the single highest-leverage improvement.
- **Cross-validate the SimpleBaseline** properly (k-fold) to get an honest
  comparison number against the LLM agent, rather than the current
  fit-on-same-data estimate.
- **Use a genuinely independent judge model/provider** to remove the
  remaining self-preference-bias risk, once budget/access allows.
- **Handle full multi-turn threads**, not just first customer message ->
  Delta reply pairs, to capture cases where severity escalates across a
  conversation.
- **Add a small set of few-shot examples specifically for sarcasm/tone
  detection** to the classifier prompt, targeting failure mode #3.
- **Re-sample the golden set with a mix of stratified AND random rows**
  to get both minority-class coverage and an honest read on true
  escalation-rate distribution.
