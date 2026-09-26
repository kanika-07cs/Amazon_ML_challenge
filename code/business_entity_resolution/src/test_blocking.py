import pandas as pd
import numpy as np
import os
import time
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_dataframe

def test_tfidf_blocking(sample_size=10000, top_k=15):
    print(f"Testing TF-IDF Blocking on {sample_size:,} S1 entities...", flush=True)
    start_time = time.time()

    # Load ground truth sample
    print("Loading ground truth...", flush=True)
    gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=sample_size, dtype=str)
    gt_df['matched_list'] = gt_df['matched_entity_ids'].fillna('').apply(
        lambda x: [i.strip() for i in str(x).split(',') if i.strip()]
    )
    s1_sample_ids = set(gt_df['source1_entity_id'])

    # Load S1 sample
    print("Loading S1 sample...", flush=True)
    s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', dtype=str)
    s1_df = s1_df[s1_df['entity_id'].isin(s1_sample_ids)].copy()
    s1_df = preprocess_dataframe(s1_df)

    # Load S2 and S3
    print("Loading S2 and S3...", flush=True)
    s2_df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', dtype=str)
    s3_df = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', dtype=str)

    s2_df = preprocess_dataframe(s2_df)
    s3_df = preprocess_dataframe(s3_df)

    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    print(f"Total Target records (S2 + S3): {len(target_df):,}", flush=True)

    s1_df['full_text'] = (s1_df['norm_name'] + " " + s1_df['norm_address']).str.strip()
    target_df['full_text'] = (target_df['norm_name'] + " " + target_df['norm_address']).str.strip()

    print("Fitting TF-IDF Vectorizer...", flush=True)
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=3, max_features=50000, dtype=np.float32)
    
    target_tfidf = vectorizer.fit_transform(target_df['full_text'])
    s1_tfidf = vectorizer.transform(s1_df['full_text'])

    print(f"TF-IDF matrices created: S1 shape={s1_tfidf.shape}, Target shape={target_tfidf.shape}", flush=True)

    chunk_size = 2000
    candidate_dict = {}
    target_ids = target_df['entity_id'].values
    s1_ids = s1_df['entity_id'].values

    total_true_matches = 0
    retained_true_matches = 0

    print("Retrieving candidates using sparse dot product...", flush=True)
    for i in range(0, len(s1_df), chunk_size):
        s1_chunk = s1_tfidf[i:i+chunk_size]
        sim_matrix = s1_chunk.dot(target_tfidf.T) # Sparse matrix multiplication

        for r_idx in range(sim_matrix.shape[0]):
            row = sim_matrix[r_idx]
            if row.nnz > 0:
                n_select = min(top_k, row.nnz)
                top_indices = row.indices[np.argpartition(row.data, -n_select)[-n_select:]]
                scores = row.data[np.argpartition(row.data, -n_select)[-n_select:]]
                sorted_order = np.argsort(-scores)
                top_indices = top_indices[sorted_order]
                cand_ids = target_ids[top_indices].tolist()
            else:
                cand_ids = []
            
            s1_id = s1_ids[i + r_idx]
            candidate_dict[s1_id] = cand_ids

        print(f"Processed {min(i+chunk_size, len(s1_df)):,}/{len(s1_df):,} S1 entities...", flush=True)

    # Measure recall against ground truth
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

    print("\n--- TF-IDF BLOCKING RESULTS ---", flush=True)
    print(f"Sample S1 count: {sample_size:,}", flush=True)
    print(f"Top K candidates per entity: {top_k}", flush=True)
    print(f"Total True Matches in Ground Truth: {total_true_matches:,}", flush=True)
    print(f"Retained True Matches in Candidates: {retained_true_matches:,}", flush=True)
    print(f"Candidate Match Recall: {recall:.2f}%", flush=True)
    print(f"Time taken: {elapsed:.2f} seconds", flush=True)

if __name__ == '__main__':
    test_tfidf_blocking(sample_size=5000, top_k=15)
