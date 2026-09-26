import pandas as pd
import numpy as np
import os
import time
import sys
import gc
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_dataframe

LEGAL_SUFFIXES = {
    'inc', 'llc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited', 'pvt', 'private',
    'gmbh', 'plc', 'srl', 'bv', 'oy', 'services', 'solutions', 'group', 'enterprises', 'holding'
}

def extract_all_keys(name, address):
    keys = []
    norm_name = str(name).lower()
    norm_addr = str(address).lower()
    
    # 1. Name tokens
    n_tokens = [t for t in norm_name.split() if len(t) >= 3 and t not in LEGAL_SUFFIXES]
    keys.extend(n_tokens)
    
    # 2. Name prefix 4-gram
    clean_n = "".join(n_tokens)
    if len(clean_n) >= 4:
        keys.append("pref_" + clean_n[:4])
        
    # 3. Address tokens
    a_tokens = [t for t in norm_addr.split() if t.isdigit() or (len(t) >= 4 and t not in LEGAL_SUFFIXES)]
    keys.extend(["addr_" + t for t in a_tokens[:3]])
    
    return list(set(keys))

def test_enriched_blocking(sample_size=3000, top_k=25, max_posting_list=5000):
    print(f"Testing Enriched Sequential Multi-Key Blocking on {sample_size:,} S1 entities...", flush=True)
    start_time = time.time()

    print("Loading ground truth...", flush=True)
    gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=sample_size, dtype=str)
    gt_df['matched_list'] = gt_df['matched_entity_ids'].fillna('').apply(
        lambda x: [i.strip() for i in str(x).split(',') if i.strip()]
    )
    s1_sample_ids = set(gt_df['source1_entity_id'])

    print("Loading & preprocessing S1 sample...", flush=True)
    s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=str)
    s1_df = s1_df[s1_df['entity_id'].isin(s1_sample_ids)].copy()
    s1_df = preprocess_dataframe(s1_df)

    country_key_index = defaultdict(lambda: defaultdict(list))
    global_key_index = defaultdict(list)

    # Process S2 into index sequentially
    print("Loading & indexing S2...", flush=True)
    t0 = time.time()
    s2_df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    s2_df = preprocess_dataframe(s2_df)
    
    for eid, ctry, name, addr in zip(s2_df['entity_id'].values, s2_df['country'].values, s2_df['norm_name'].values, s2_df['norm_address'].values):
        keys = extract_all_keys(name, addr)
        for k in keys:
            if len(country_key_index[ctry][k]) < max_posting_list:
                country_key_index[ctry][k].append(eid)
            if len(global_key_index[k]) < max_posting_list:
                global_key_index[k].append(eid)

    print(f"Indexed S2 ({len(s2_df):,} records) in {time.time() - t0:.2f}s", flush=True)
    del s2_df
    gc.collect()

    # Process S3 into index sequentially
    print("Loading & indexing S3...", flush=True)
    t0 = time.time()
    s3_df = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    s3_df = preprocess_dataframe(s3_df)

    for eid, ctry, name, addr in zip(s3_df['entity_id'].values, s3_df['country'].values, s3_df['norm_name'].values, s3_df['norm_address'].values):
        keys = extract_all_keys(name, addr)
        for k in keys:
            if len(country_key_index[ctry][k]) < max_posting_list:
                country_key_index[ctry][k].append(eid)
            if len(global_key_index[k]) < max_posting_list:
                global_key_index[k].append(eid)

    print(f"Indexed S3 ({len(s3_df):,} records) in {time.time() - t0:.2f}s", flush=True)
    del s3_df
    gc.collect()

    print("Retrieving candidates for S1 sample...", flush=True)
    ret_t0 = time.time()
    candidate_dict = {}
    
    s1_eids = s1_df['entity_id'].values
    s1_countries = s1_df['country'].values
    s1_names = s1_df['norm_name'].values
    s1_addrs = s1_df['norm_address'].values
    
    for s1_id, ctry, name, addr in zip(s1_eids, s1_countries, s1_names, s1_addrs):
        keys = extract_all_keys(name, addr)
        cand_counts = defaultdict(int)
        
        c_idx = country_key_index.get(ctry, global_key_index)
        for k in keys:
            if k in c_idx:
                for cand_id in c_idx[k]:
                    cand_counts[cand_id] += 1
                    
        if len(cand_counts) < top_k:
            for k in keys:
                if k in global_key_index:
                    for cand_id in global_key_index[k]:
                        cand_counts[cand_id] += 1
                        
        if cand_counts:
            top_cands = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            candidate_dict[s1_id] = [c[0] for c in top_cands]
        else:
            candidate_dict[s1_id] = []

    print(f"Candidate retrieval finished in {time.time() - ret_t0:.2f}s", flush=True)

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

    print("\n--- ENRICHED MULTI-KEY BLOCKING RESULTS ---", flush=True)
    print(f"Sample S1 count: {sample_size:,}", flush=True)
    print(f"Top K candidates per entity: {top_k}", flush=True)
    print(f"Total True Matches in Ground Truth: {total_true_matches:,}", flush=True)
    print(f"Retained True Matches in Candidates: {retained_true_matches:,}", flush=True)
    print(f"Candidate Match Recall: {recall:.2f}%", flush=True)
    print(f"Total time taken: {elapsed:.2f} seconds", flush=True)

if __name__ == '__main__':
    test_enriched_blocking(sample_size=3000, top_k=25, max_posting_list=5000)
