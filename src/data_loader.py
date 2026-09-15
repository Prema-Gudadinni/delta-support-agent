"""
data_loader.py

Responsible for:
1. Loading the filtered Delta thread data (delta_threads.csv).
2. Reconstructing (customer_message -> delta_reply) PAIRS -- this is the
   basic unit our whole system works on.
3. Splitting those pairs into:
     - a GROUNDING POOL (used by the reply drafter to find similar past
       resolutions)
     - the GOLDEN SET tweet_ids, which are EXCLUDED from the grounding pool.

WHY the split matters (read this before you touch anything else):
If a golden-set customer message's own historical reply is sitting in the
grounding pool, the drafter can just copy/paraphrase the "answer key" instead
of actually generalizing. Your eval numbers would then be inflated and not
represent real performance on unseen messages. This is the single most
common way a "headline number" ends up misleading -- so we design it out
from the start rather than discovering it later.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
THREADS_PATH = DATA_DIR / "delta_threads.csv"
GOLDEN_SET_PATH = DATA_DIR / "golden_set_labeled.csv"  # produced by YOU, after hand-labeling


def load_raw_threads() -> pd.DataFrame:
    """Load the filtered Delta thread rows (both customer + Delta messages)."""
    df = pd.read_csv(THREADS_PATH, dtype=str)
    df["text"] = df["text"].str.replace("\n", " ", regex=False)
    return df


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct (customer_message, delta_reply) pairs.

    Logic: for every Delta reply, find the customer tweet it was
    'in_response_to'. That gives us one row per resolved exchange.

    NOTE: this only captures the FIRST Delta reply to each customer message
    (not full multi-turn back-and-forth). That's a deliberate scope
    decision -- log it in your decision log. Multi-turn threads are a
    natural "next week" extension.
    """
    delta_replies = df[df["author_id"] == "Delta"].copy()
    customer_msgs = df[df["inbound"] == "True"].copy()

    customer_lookup = customer_msgs.set_index("tweet_id")["text"].to_dict()

    pairs = []
    for row in delta_replies.itertuples():
        cust_id = row.in_response_to_tweet_id
        if pd.isna(cust_id) or cust_id not in customer_lookup:
            continue
        pairs.append({
            "customer_tweet_id": cust_id,
            "customer_text": customer_lookup[cust_id],
            "delta_reply_tweet_id": row.tweet_id,
            "delta_reply_text": row.text,
        })

    return pd.DataFrame(pairs).drop_duplicates(subset="customer_tweet_id")


def get_golden_set_tweet_ids() -> set:
    """
    Returns the tweet_ids that are reserved for evaluation, so they can be
    excluded from the grounding pool. Reads from the labeling template
    (works even before you've filled in labels -- we only need the IDs).
    """
    template_path = DATA_DIR / "golden_set_to_label.csv"
    if not template_path.exists():
        return set()
    ids = pd.read_csv(template_path, dtype=str)["tweet_id"]
    return set(ids)


def get_grounding_pool() -> pd.DataFrame:
    """
    The pairs the reply drafter is ALLOWED to look at and learn from.
    Excludes anything reserved for the golden set.
    """
    df = load_raw_threads()
    pairs = build_pairs(df)
    golden_ids = get_golden_set_tweet_ids()
    pool = pairs[~pairs["customer_tweet_id"].isin(golden_ids)]
    return pool.reset_index(drop=True)


def load_golden_set() -> pd.DataFrame:
    """
    Loads YOUR hand-labeled golden set. Will raise a clear error if you
    haven't finished labeling yet, so we fail loudly instead of silently
    evaluating against empty/garbage labels.
    """
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(
            f"{GOLDEN_SET_PATH} not found. Finish hand-labeling "
            f"data/golden_set_to_label.csv, then save it as "
            f"data/golden_set_labeled.csv before running eval."
        )
    df = pd.read_csv(GOLDEN_SET_PATH, dtype=str)
    required_cols = {"tweet_id", "text", "your_intent_label", "should_escalate (yes/no)"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Golden set is missing columns: {missing}")
    unlabeled = df["your_intent_label"].isna() | (df["your_intent_label"].str.strip() == "")
    if unlabeled.any():
        raise ValueError(f"{unlabeled.sum()} rows still have no intent label. Finish labeling first.")
    return df


if __name__ == "__main__":
    # Quick sanity check you can run any time with: python3 src/data_loader.py
    pool = get_grounding_pool()
    golden_ids = get_golden_set_tweet_ids()
    print(f"Grounding pool size (leakage-safe): {len(pool)} pairs")
    print(f"Golden set reserved IDs: {len(golden_ids)}")
    overlap = set(pool["customer_tweet_id"]) & golden_ids
    print(f"Overlap check (should be 0): {len(overlap)}")
