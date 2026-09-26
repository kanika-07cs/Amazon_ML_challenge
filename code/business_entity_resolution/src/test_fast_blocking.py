import pandas as pd
import numpy as np
import os
import time
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_dataframe, extract_blocking_tokens

COMMON_STOP_TOKENS = {
    'inc', 'llc', 'corp', 'corporation', 'company', 'ltd', 'limited', 'pvt', 'private',
    'gmbh', 'plc', 'co', 'services', 'solutions', 'group', 'enterprises', 'holding',
    'holdings', 'international', 'global', 'center', 'centre', 'retail', 'tech',
    'technologies', 'store', 'shop', 'mart', 'supermarket', 'traders', 'trading',
    'india', 'usa', 'america', 'north', 'south', 'east', 'west', 'city', 'state',
    'hotel', 'restaurant', 'pharmacy', 'medical', 'hospital', 'school', 'college',
    'institute', 'agency', 'associate', 'associates', 'industries', 'industry'
}

def test_fast_country_token_blocking(sample_size=10000, top_k=20, max_posting_list=2000):
    print(f"Testing High-Speed Filtered Blocking on {sample_size:,} S1 entities...", flush=True)
    start_time = time.time()

    print("Loading ground truth...", flush=True)
    t0 = time.time()
    gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=sample_size, dtype=str)
    gt_df['matched_list'] = gt_df['matched_entity_ids'].fillna('').apply(
        lambda x: [i.strip() for i in str(x).split(',') if i.strip()]
    )
    s1_sample_ids = set(gt_df['source1_entity_id'])
    print(f"Loaded ground truth in {time.time() - t0:.2f}s", flush=True)

    print("Loading S1 sample...", flush=True)
    t0 = time.time()
    s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=str)
    s1_df = s1_df[s1_df['entity_id'].isin(s1_sample_ids)].copy()
    s1_df = preprocess_dataframe(s1_df)
    print(f"Loaded & preprocessed S1 in {time.time() - t0:.2f}s", flush=True)

    print("Loading S2 and S3...", flush=True)
    t0 = time.time()
    s2_df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    s3_df = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    print(f"Loaded TSVs in {time.time() - t0:.2f}s", flush=True)

    print("Preprocessing S2 and S3...", flush=True)
    t0 = time.time()
    s2_df = preprocess_dataframe(s2_df)
    s3_df = preprocess_dataframe(s3_df)
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    print(f"Preprocessed {len(target_df):,} target records in {time.time() - t0:.2f}s", flush=True)

    print("Indexing target records with posting list cap...", flush=True)
    idx_start = time.time()
    
    country_token_index = defaultdict(lambda: defaultdict(list))
    global_token_index = defaultdict(list)
    
    eids = target_df['entity_id'].values
    countries = target_df['country'].values
    names = target_df['norm_name'].values
    
    for eid, ctry, norm_name in zip(eids, countries, names):
        tokens = norm_name.split()
        if not tokens:
            continue
        # Filter out common stop tokens and length < 3
        valid_tokens = [t for t in set(tokens) if len(t) >= 3 and t not in COMMON_STOP_TOKENS]
        for tok in valid_tokens[:3]:
            if len(country_token_index[ctry][tok]) < max_posting_list:
                country_token_index[ctry][tok].append(eid)
            if len(global_token_index[tok]) < max_posting_list:
                global_token_index[tok].append(eid)

    print(f"Index created in {time.time() - idx_start:.2f}s. Indexed countries: {len(country_token_index):,}", flush=True)

    print("Retrieving candidates for S1 sample...", flush=True)
    retrieval_start = time.time()
    candidate_dict = {}
    
    s1_eids = s1_df['entity_id'].values
    s1_countries = s1_df['country'].values
    s1_names = s1_df['norm_name'].values
    
    for s1_id, ctry, norm_name in zip(s1_eids, s1_countries, s1_names):
        tokens = norm_name.split()
        valid_tokens = [t for t in set(tokens) if len(t) >= 3 and t not in COMMON_STOP_TOKENS]
        if not valid_tokens:
            valid_tokens = [t for t in set(tokens) if len(t) >= 2]

        cand_counts = defaultdict(int)
        c_idx = country_token_index.get(ctry, global_token_index)
        
        for tok in valid_tokens[:4]:
            if tok in c_idx:
                for cand_id in c_idx[tok]:
                    cand_counts[cand_id] += 1
        
        if len(cand_counts) < top_k:
            for tok in valid_tokens[:3]:
                if tok in global_token_index:
                    for cand_id in global_token_index[tok]:
                        cand_counts[cand_id] += 1
                        
        if cand_counts:
            top_cands = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            candidate_dict[s1_id] = [c[0] for c in top_cands]
        else:
            candidate_dict[s1_id] = []

    print(f"Candidate retrieval completed in {time.time() - retrieval_start:.2f}s!", flush=True)

    # Measure recall
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

    print("\n--- HIGH-SPEED FILTERED BLOCKING RESULTS ---", flush=True)
    print(f"Sample S1 count: {sample_size:,}", flush=True)
    print(f"Top K candidates per entity: {top_k}", flush=True)
    print(f"Max Posting List Cap per Token: {max_posting_list:,}", flush=True)
    print(f"Total True Matches in Ground Truth: {total_true_matches:,}", flush=True)
    print(f"Retained True Matches in Candidates: {retained_true_matches:,}", flush=True)
    print(f"Candidate Match Recall: {recall:.2f}%", flush=True)
    print(f"Total time taken: {elapsed:.2f} seconds", flush=True)

if __name__ == '__main__':
    test_fast_country_token_blocking(sample_size=10000, top_k=20, max_posting_list=2000)
