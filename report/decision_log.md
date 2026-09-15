# Decision Log

Non-obvious decisions made while building this, and why. (Add more as you
label the golden set and run evals -- you'll hit more edge cases.)

1. **Chose Delta over higher-volume brands (Amazon, Apple).** Airlines have
   concrete, groundable resolution content (rebooking, vouchers, baggage
   procedures) vs. generic "please DM us" replies common in Amazon/Apple
   threads. Traded some volume for reply-quality groundability.

2. **Discovered and corrected an Excel truncation bug before it corrupted
   analysis.** Initial brand-volume counts came from a pivot table in Excel,
   which silently truncates at 1,048,576 rows. Real dataset has 2.81M rows --
   Delta's true tweet count was 42,253, not the 16,372 Excel showed (2.6x
   undercount). Caught by cross-checking with a full pandas load.

3. **Intent taxonomy derived from reading real data, not assumed.** Read two
   random samples (35 + 30 = 65 messages) of actual Delta customer messages
   before defining categories. Added `loyalty_miles_status` and
   `praise_or_noise` only after seeing them recur in the second sample --
   the first sample alone wouldn't have surfaced them.

4. **`praise_or_noise` is a deliberate catch-category, not a cop-out.**
   ~20% of sampled messages were compliments or off-topic mentions with no
   actionable request. Forcing these into "real" support intents would
   inflate apparent classifier accuracy artificially (they're trivially
   easy to classify) while hiding that the model isn't being tested on
   what matters.

5. **Escalation is a separate decision layer from intent, not derived from
   it.** A `refund_compensation` message can be routine or a genuine crisis
   (stranded overnight) -- severity and intent are orthogonal. Modeled as
   two independent decisions in the pipeline.

6. **Escalation uses a rule-based safety net for high-stakes categories
   (injury/legal/discrimination), with LLM judgment for everything else.**
   Pure LLM judgment felt too risky for the highest-cost failure mode
   (missing a real safety issue); pure keyword rules would miss creative
   phrasing for lower-stakes cases. Hybrid trades some elegance for an
   auditable guardrail on the cases that matter most.

7. **Golden set is provably excluded from the reply-drafter's grounding
   pool.** Verified via an automated overlap check (0 overlap) rather than
   just asserting it -- data leakage here would silently inflate reply
   quality scores.

8. **Reply retrieval uses TF-IDF cosine similarity, not embeddings.**
   Faster, free, and transparent about *why* an example was retrieved
   (shared vocabulary). Trade-off: misses semantically similar messages
   that use different words (documented as a "next week" improvement).

9. **Switched from Anthropic API → Google Gemini API → Groq API, in that
    order, purely for access/cost reasons under a tight deadline.**
    - Anthropic: required paid credits, not viable with no budget.
    - Gemini (free tier): burned significant time on what looked like
      model-availability 404s across multiple model names, eventually
      traced to an auth-method issue (key needed to be passed as a
      `?key=` query parameter, not an `x-goog-api-key` header) -- and
      even after fixing that, hit further friction with a newer
      AI-Studio key format. Given the deadline, stopped debugging further.
    - Groq (free tier): standard Bearer-token auth, OpenAI-compatible
      request shape, no phone/card verification, worked immediately.
    Agent and judge use genuinely different model FAMILIES on Groq
    (openai/gpt-oss-20b vs qwen/qwen3.6-27b -- the originally-chosen
    llama-3.1-8b-instant / llama-3.3-70b-versatile pair turned out to have
    been deprecated by Groq in August 2026, after this project's model
    knowledge was formed -- caught via a direct curl call returning
    Groq's actual `model_not_found` error, then confirmed via Groq's own
    deprecation notes and migration guidance), which is a stronger
    self-preference-bias mitigation than the same-provider,
    different-tier setup attempted earlier with Gemini/Claude.
    Lesson worth keeping in the log either way: a string of "not found"
    errors across DIFFERENT model names on the same provider was a
    signal to question the request mechanics (auth method), not keep
    guessing model names -- useful debugging instinct beyond this project.

9c. **Simplified back to a SINGLE model (gemini-2.0-flash-lite) for both
    agent and judge, after the auth fix in #9b made a two-model setup
    possible again.** Chose reliability and simplicity over the bias
    mitigation, given free-tier rate-limit pressure and the Sept 17
    deadline. CONSEQUENCE: no self-preference-bias mitigation -- the same
    model that drafts replies also grades them, which likely inflates
    reply-quality scores somewhat. This is explicitly named in report.md
    section 5 ("what's misleading about my headline number"), not hidden.
    A natural "one more week" improvement would be reintroducing a
    genuinely separate judge model/provider.

10. **Golden set stratified-sampled by heuristic keyword bucket, not
    pure random.** Pure random sampling from the raw distribution would
    have made rare-but-high-stakes intents (rebooking, refund) nearly
    invisible in a 216-row set. Stratifying protects minority classes at
    the cost of not perfectly reflecting real-world frequency -- documented
    and correctable via reweighting if needed.

11. **[Fill in during labeling]** -- decisions about ambiguous/multi-intent
    messages, e.g. how you resolved cases like "flight delayed AND crew was
    rude" (#9 from our sample review).

12. **[Fill in after eval]** -- any decisions made while debugging the
    classifier/judge (e.g. prompt adjustments after seeing failure patterns).

13-15. **[To be added]**
