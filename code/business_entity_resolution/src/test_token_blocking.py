import pandas as pd
import numpy as np
import os
import time
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_dataframe, extract_blocking_tokens

def test_token_blocking(sample_size=10000, top_k=20):
    print(f"Testing Fast Inverted Token Blocking on {sample_size:,} S1 entities...", flush=True)
    start_time = time.time()

    print("Loading ground truth...", flush=True)
    gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=sample_size, dtype=str)
    gt_df['matched_list'] = gt_df['matched_entity_ids'].fillna('').apply(
        lambda x: [i.strip() for i in str(x).split(',') if i.strip()]
    )
    s1_sample_ids = set(gt_df['source1_entity_id'])

    print("Loading S1 sample...", flush=True)
    s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=str)
    s1_df = s1_df[s1_df['entity_id'].isin(s1_sample_ids)].copy()
    s1_df = preprocess_dataframe(s1_df)

    print("Loading S2 and S3...", flush=True)
    s2_df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    s3_df = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)

    s2_df = preprocess_dataframe(s2_df)
    s3_df = preprocess_dataframe(s3_df)

    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    print(f"Total Target records (S2 + S3): {len(target_df):,}", flush=True)

    print("Building Fast Inverted Token Index...", flush=True)
    index_start = time.time()
    token_to_target_ids = defaultdict(list)
    
    target_names = target_df['business_name'].values
    target_ids = target_df['entity_id'].values
    
    for eid, name in zip(target_ids, target_names):
        tokens = extract_blocking_tokens(name)
        for tok in set(tokens):
            token_to_target_ids[tok].append(eid)
    
    print(f"Inverted Index built in {time.time() - index_start:.2f} seconds. Unique tokens: {len(token_to_target_ids):,}", flush=True)

    print("Querying index for S1 sample candidates...", flush=True)
    candidate_dict = {}
    s1_names = s1_df['business_name'].values
    s1_ids = s1_df['entity_id'].values
    
    for s1_id, name in zip(s1_ids, s1_names):
        tokens = extract_blocking_tokens(name)
        candidate_counts = defaultdict(int)
        for tok in set(tokens):
            for cand_id in token_to_target_ids.get(tok, []):
                candidate_counts[cand_id] += 1
        
        if candidate_counts:
            # Get top_k by frequency
            top_cands = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            cand_ids = [c[0] for c in top_cands]
        else:
            cand_ids = []
            
        candidate_dict[s1_id] = cand_ids

    # Measure recall against ground truth
    total_true_matches = 0
    retained_true_matches = 0
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        true_matches = row['matched_list']
        if not true_matches:
            continue
        
        total_true_matches += len(true_matches)
        cands = set(candidate_dict.get(s1_id, []))
        retained = sum(1 for m in true_matches if m in cands)
        retained_true_matches += retained

    recall = (retained_true_matches / total_true_matches) * 100 if total_true_matches > 0 else 0.0
    elapsed = time.time() - start_time

    print("\n--- INVERTED TOKEN BLOCKING RESULTS ---", flush=True)
    print(f"Sample S1 count: {sample_size:,}", flush=True)
    print(f"Top K candidates per entity: {top_k}", flush=True)
    print(f"Total True Matches in Ground Truth: {total_true_matches:,}", flush=True)
    print(f"Retained True Matches in Candidates: {retained_true_matches:,}", flush=True)
    print(f"Candidate Match Recall: {recall:.2f}%", flush=True)
    print(f"Total time taken: {elapsed:.2f} seconds", flush=True)

if __name__ == '__main__':
    test_token_blocking(sample_size=10000, top_k=20)
