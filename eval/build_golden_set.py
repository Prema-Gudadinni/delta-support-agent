import pandas as pd
import re

df = pd.read_csv('../delta_threads.csv', dtype=str)
customer_msgs = df[df['inbound'] == 'True'].copy()
customer_msgs['text'] = customer_msgs['text'].str.replace('\n', ' ', regex=False)

# Lightweight heuristic pre-tagging just to help STRATIFY sampling across our
# draft buckets, so the golden set isn't accidentally 80% "flight delay."
# This is NOT the classifier -- purely a sampling aid, and will be thrown away
# after we pull the sample. Final labels are 100% hand-done.
patterns = {
    'flight_delay_status': r'\b(delay|delayed|late|mechanical|waiting|stuck)\b',
    'rebooking_connection': r'\b(rebook|connect|connection|miss(ed)? (my|the) flight|alternate)\b',
    'refund_compensation': r'\b(refund|voucher|compensat|credit|reimburse)\b',
    'checkin_seat_upgrade': r'\b(check.?in|upgrade|seat)\b',
    'baggage_lost_item': r'\b(bag|baggage|luggage|carousel|lost (my|item))\b',
    'loyalty_miles_status': r'\b(mile|mqm|mqd|mqs|medallion|diamond|skymiles|status)\b',
    'service_complaint': r'\b(rude|terrible|worst|awful|horrible|complain)\b',
    'praise_noise': r'\b(thank|thanks|great|awesome|love|amazing)\b',
}

def tag(text):
    t = text.lower()
    for label, pat in patterns.items():
        if re.search(pat, t):
            return label
    return 'general_policy_info_or_other'

customer_msgs['heuristic_bucket'] = customer_msgs['text'].apply(tag)
print(customer_msgs['heuristic_bucket'].value_counts())

# Stratified sample: aim for ~22-25 per bucket (8 buckets) = ~180-200 total,
# capped by bucket availability
target_per_bucket = 24
samples = []
for bucket, group in customer_msgs.groupby('heuristic_bucket'):
    n = min(target_per_bucket, len(group))
    samples.append(group.sample(n=n, random_state=7))

golden_candidates = pd.concat(samples).sample(frac=1, random_state=7).reset_index(drop=True)
print(f"\nTotal golden set candidates pulled: {len(golden_candidates)}")

# Build labeling template
labeling_df = golden_candidates[['tweet_id', 'text', 'in_response_to_tweet_id', 'heuristic_bucket']].copy()
labeling_df = labeling_df.rename(columns={'heuristic_bucket': 'suggested_bucket_IGNORE_ME'})
labeling_df['your_intent_label'] = ''
labeling_df['should_escalate (yes/no)'] = ''
labeling_df['escalation_reason'] = ''
labeling_df['notes'] = ''

labeling_df.to_csv('data/golden_set_to_label.csv', index=False)
print("Saved data/golden_set_to_label.csv")
