"""
reply_drafter.py

Drafts a reply to a customer message, GROUNDED in how Delta has actually
replied to similar past messages (deliverable #2).

Approach: TF-IDF similarity retrieval (not embeddings) over the leakage-safe
grounding pool (src/data_loader.get_grounding_pool) -- simple, fast, and
transparent about *why* a given example was retrieved. This is a documented
scope decision: an embedding-based retriever would likely find better
semantic matches (e.g. "my flight got pushed back" vs "delayed" without
shared vocabulary) -- flagged as a "what I'd do with one more week" item.

We fit the TF-IDF vectorizer ONCE on the grounding pool and reuse it, rather
than re-fitting per call, for speed and consistency.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.llm_client import call_agent
from src.data_loader import get_grounding_pool

_vectorizer = None
_pool_vectors = None
_pool_df = None


def _ensure_index():
    global _vectorizer, _pool_vectors, _pool_df
    if _vectorizer is None:
        _pool_df = get_grounding_pool()
        _vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        _pool_vectors = _vectorizer.fit_transform(_pool_df["customer_text"])


def retrieve_similar(message: str, k: int = 3):
    """Returns the top-k most similar (customer_text, delta_reply_text) pairs
    from the grounding pool, using TF-IDF cosine similarity."""
    _ensure_index()
    query_vec = _vectorizer.transform([message])
    sims = cosine_similarity(query_vec, _pool_vectors)[0]
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        results.append({
            "similar_customer_text": _pool_df.iloc[i]["customer_text"],
            "delta_reply": _pool_df.iloc[i]["delta_reply_text"],
            "similarity": float(sims[i]),
        })
    return results


DRAFT_PROMPT_TEMPLATE = """Delta Air Lines Twitter support. Past examples:

{examples}

Draft a reply to this NEW message, same tone/style, concise, don't invent facts not in the message.

Message: "{message}"

Reply:"""


def draft_reply(message: str, k: int = 2) -> dict:
    # k reduced from 3 to 2, and example text truncated, to cut prompt
    # tokens under Groq's free-tier TPM cap -- logged as a real tradeoff
    # (less grounding context per draft) in the decision log, not silent.
    similar = retrieve_similar(message, k=k)
    examples_text = "\n\n".join(
        f'Customer: "{s["similar_customer_text"][:150]}"\nDelta: "{s["delta_reply"][:150]}"'
        for s in similar
    )
    prompt = DRAFT_PROMPT_TEMPLATE.format(examples=examples_text, message=message)
    reply = call_agent(prompt, max_tokens=120).strip()
    return {
        "reply": reply,
        "grounding_examples": similar,  # kept for failure analysis / transparency
    }
