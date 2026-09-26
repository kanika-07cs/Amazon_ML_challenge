import time
import gc
import heapq
from collections import defaultdict
import pandas as pd
import numpy as np
from preprocess import preprocess_dataframe

LEGAL_SUFFIXES = {
    'inc', 'llc', 'corp', 'corporation', 'co', 'company', 'ltd', 'limited', 'pvt', 'private',
    'gmbh', 'plc', 'srl', 'bv', 'oy', 'services', 'solutions', 'group', 'enterprises', 'holding',
    'holdings', 'international', 'global', 'center', 'centre', 'retail', 'tech', 'technologies',
    'store', 'shop', 'mart', 'supermarket', 'traders', 'trading', 'industries', 'industry'
}

def extract_record_keys(name, address):
    """
    Extract multi-aspect blocking keys for entity resolution:
    - Distinctive name tokens (length >= 3, non-legal)
    - 3-gram character prefixes of name tokens
    - Bigrams of consecutive name tokens
    - Address tokens (digits or length >= 4)
    """
    keys = []
    norm_name = str(name).lower()
    norm_addr = str(address).lower()
    
    n_tokens = [t for t in norm_name.split() if len(t) >= 3 and t not in LEGAL_SUFFIXES]
    
    # 1. Unigram name tokens
    keys.extend(n_tokens)
    
    # 2. 3-gram character prefixes of name tokens
    for t in n_tokens:
        if len(t) >= 4:
            keys.append("p3_" + t[:3])
            
    # 3. Bigrams of consecutive name tokens
    if len(n_tokens) >= 2:
        for i in range(len(n_tokens) - 1):
            keys.append("bg_" + n_tokens[i] + "_" + n_tokens[i+1])
            
    # 4. Address tokens (digits or length >= 4)
    a_tokens = [t for t in norm_addr.split() if t.isdigit() or (len(t) >= 4 and t not in LEGAL_SUFFIXES)]
    keys.extend(["addr_" + t for t in a_tokens[:3]])
    
    return list(set(keys))

def build_candidate_index(s2_df, s3_df, max_posting_list=3000):
    """
    Build a memory-efficient inverted index mapping blocking keys to target entity IDs.
    Indexed by country to keep memory footprint low and lookups sub-millisecond.
    """
    print("Building Memory-Optimized Candidate Blocking Index...", flush=True)
    t0 = time.time()
    
    country_key_index = defaultdict(lambda: defaultdict(list))

    for df, name in [(s2_df, "Source 2"), (s3_df, "Source 3")]:
        eids = df['entity_id'].values
        countries = df['country'].values
        names = df['norm_name'].values
        addrs = df['norm_address'].values
        
        for eid, ctry, n, a in zip(eids, countries, names, addrs):
            keys = extract_record_keys(n, a)
            ctry_key = ctry if ctry else 'GLOBAL'
            for k in keys:
                if len(country_key_index[ctry_key][k]) < max_posting_list:
                    country_key_index[ctry_key][k].append(eid)
                    
    print(f"Index built successfully in {time.time() - t0:.2f} seconds. Indexed countries: {list(country_key_index.keys())}", flush=True)
    return country_key_index

def generate_candidates_for_s1(s1_df, country_key_index, top_k=30):
    """
    Retrieve top-K candidate target IDs for each S1 entity.
    Supports open-set unseen countries (e.g. France in test set) via cross-country fallback.
    Returns dict: s1_entity_id -> list of candidate entity IDs.
    """
    print(f"Retrieving top-{top_k} candidates for {len(s1_df):,} S1 entities...", flush=True)
    t0 = time.time()
    
    candidate_dict = {}
    s1_eids = s1_df['entity_id'].values
    s1_countries = s1_df['country'].values
    s1_names = s1_df['norm_name'].values
    s1_addrs = s1_df['norm_address'].values

    all_indexed_countries = list(country_key_index.keys())

    for s1_id, ctry, n, a in zip(s1_eids, s1_countries, s1_names, s1_addrs):
        keys = extract_record_keys(n, a)
        cand_counts = defaultdict(int)
        
        # Primary lookup: target country index
        target_ctries = [ctry] if ctry in country_key_index else all_indexed_countries
        
        for target_c in target_ctries:
            c_idx = country_key_index[target_c]
            for k in keys:
                if k in c_idx:
                    for cand_id in c_idx[k]:
                        cand_counts[cand_id] += 1
                        
        # Fallback to all countries if fewer than top_k candidates found for an unseen country
        if len(cand_counts) < top_k and ctry not in country_key_index:
            for fallback_c in all_indexed_countries:
                if fallback_c not in target_ctries:
                    c_idx = country_key_index[fallback_c]
                    for k in keys:
                        if k in c_idx:
                            for cand_id in c_idx[k]:
                                cand_counts[cand_id] += 1
                                
        if cand_counts:
            # Use heapq.nlargest for fast top-K candidate extraction (O(N log K) instead of O(N log N))
            top_cands = heapq.nlargest(top_k, cand_counts.items(), key=lambda x: x[1])
            candidate_dict[s1_id] = [c[0] for c in top_cands]
        else:
            candidate_dict[s1_id] = []
            
    print(f"Candidate generation completed in {time.time() - t0:.2f} seconds.", flush=True)
    return candidate_dict

def evaluate_blocking_recall(candidate_dict, gt_df):
    """
    Measure candidate recall against ground truth matching pairs.
    """
    total_true_matches = 0
    retained_true_matches = 0
    
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        m_str = str(row['matched_entity_ids']) if pd.notnull(row['matched_entity_ids']) else ''
        true_matches = [m.strip() for m in m_str.split(',') if m.strip()]
        
        if not true_matches:
            continue
            
        total_true_matches += len(true_matches)
        cands = set(candidate_dict.get(s1_id, []))
        retained = sum(1 for m in true_matches if m in cands)
        retained_true_matches += retained
        
    recall = (retained_true_matches / total_true_matches) * 100 if total_true_matches > 0 else 0.0
    print("\n--- CANDIDATE BLOCKING RECALL EVALUATION ---")
    print(f"Total True Ground Truth Matches: {total_true_matches:,}")
    print(f"Retained True Matches in Candidate Pool: {retained_true_matches:,}")
    print(f"Candidate Recall: {recall:.2f}%")
    return recall
