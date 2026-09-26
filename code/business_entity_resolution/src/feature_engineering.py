import numpy as np
import pandas as pd
from rapidfuzz import fuzz

def compute_jaccard_similarity(str1, str2):
    set1 = set(str1.split())
    set2 = set(str2.split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / float(union)

def extract_pairwise_features(s1_names, s1_addrs, s1_ctries, target_names, target_addrs, target_ctries, candidate_pairs):
    """
    Generate comprehensive string matching features for candidate pairs using ultra-fast flat dict lookups.
    """
    feature_list = []
    
    feature_names = [
        'name_ratio',
        'name_partial_ratio',
        'name_token_sort_ratio',
        'name_token_set_ratio',
        'name_jaccard',
        'name_exact_match',
        'address_ratio',
        'address_partial_ratio',
        'address_token_sort_ratio',
        'address_jaccard',
        'address_exact_match',
        'country_match',
        'name_len_diff',
        'address_len_diff',
        'source_is_s2',
        'candidate_rank'
    ]

    for s1_id, cand_id, rank in candidate_pairs:
        n1 = s1_names.get(s1_id, '')
        a1 = s1_addrs.get(s1_id, '')
        c1 = s1_ctries.get(s1_id, '')

        n2 = target_names.get(cand_id, '')
        a2 = target_addrs.get(cand_id, '')
        c2 = target_ctries.get(cand_id, '')

        # 1. Name similarities
        n_ratio = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        n_p_ratio = fuzz.partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        n_ts_ratio = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        n_tset_ratio = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
        n_jaccard = compute_jaccard_similarity(n1, n2)
        n_exact = 1.0 if (n1 and n1 == n2) else 0.0

        # 2. Address similarities
        a_ratio = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        a_p_ratio = fuzz.partial_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        a_ts_ratio = fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
        a_jaccard = compute_jaccard_similarity(a1, a2)
        a_exact = 1.0 if (a1 and a1 == a2) else 0.0

        # 3. Categorical & scalar features
        c_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0
        n_len_diff = abs(len(n1) - len(n2))
        a_len_diff = abs(len(a1) - len(a2))
        is_s2 = 1.0 if str(cand_id).startswith('S2-') else 0.0

        feat_row = [
            n_ratio,
            n_p_ratio,
            n_ts_ratio,
            n_tset_ratio,
            n_jaccard,
            n_exact,
            a_ratio,
            a_p_ratio,
            a_ts_ratio,
            a_jaccard,
            a_exact,
            c_match,
            float(n_len_diff),
            float(a_len_diff),
            is_s2,
            float(rank)
        ]
        feature_list.append(feat_row)

    return np.array(feature_list, dtype=np.float32), feature_names
