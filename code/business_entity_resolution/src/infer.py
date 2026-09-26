import os
import sys
import time
import joblib
import gc
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_dataframe
from blocking import build_candidate_index, generate_candidates_for_s1
from feature_engineering import extract_pairwise_features

def run_inference(test_dir='dataset/test', model_path='model/lgbm_model.pkl', output_dir='output', top_k=25, batch_size=100000, sample_size=None):
    print("=" * 60)
    print("STAGE 8 & 9: TEST INFERENCE AND STREAMING OUTPUT GENERATION")
    print("=" * 60)

    # 1. Load trained model artifact
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Train the model first.")
        sys.exit(1)

    print(f"\n1. Loading trained model from {model_path}...", flush=True)
    saved_obj = joblib.load(model_path)
    model = saved_obj['model']
    threshold = saved_obj.get('threshold', 0.5)
    print(f"Loaded model. Applied classification threshold tau = {threshold:.2f}")

    # 2. Load test S1, S2, S3
    print("\n2. Loading Test Datasets...", flush=True)
    t0 = time.time()
    test_s1_path = os.path.join(test_dir, 'test_source1.tsv')
    test_s2_path = os.path.join(test_dir, 'test_source2.tsv')
    test_s3_path = os.path.join(test_dir, 'test_source3.tsv')

    test_s1 = pd.read_csv(test_s1_path, sep='\t', dtype=str)
    if sample_size and sample_size < len(test_s1):
        print(f"Sampling Test S1 to top {sample_size:,} entities...")
        test_s1 = test_s1.head(sample_size)

    test_s2 = pd.read_csv(test_s2_path, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    test_s3 = pd.read_csv(test_s3_path, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)

    print(f"Loaded Test S1: {len(test_s1):,} rows")
    print(f"Loaded Test S2: {len(test_s2):,} rows")
    print(f"Loaded Test S3: {len(test_s3):,} rows")
    print(f"Dataset loaded in {time.time() - t0:.2f}s")

    # 3. Preprocess Test Data
    print("\n3. Preprocessing Test Data...", flush=True)
    t0 = time.time()
    test_s1 = preprocess_dataframe(test_s1)
    test_s2 = preprocess_dataframe(test_s2)
    test_s3 = preprocess_dataframe(test_s3)
    print(f"Preprocessed test data in {time.time() - t0:.2f}s")

    # 4. Build Candidate Index
    print("\n4. Building Candidate Index on Test Targets (S2 + S3)...", flush=True)
    c_index = build_candidate_index(test_s2, test_s3, max_posting_list=3000)

    # Fast Record Lookups for Target Records
    print("\n5. Preparing Target Lookups...", flush=True)
    t0 = time.time()
    target_df = pd.concat([test_s2, test_s3], ignore_index=True)
    t_eids = target_df['entity_id'].values
    target_names = dict(zip(t_eids, target_df['norm_name'].values))
    target_addrs = dict(zip(t_eids, target_df['norm_address'].values))
    target_ctries = dict(zip(t_eids, target_df['country'].values))
    del test_s2, test_s3, target_df
    gc.collect()
    print(f"Target lookups created in {time.time() - t0:.2f}s", flush=True)

    # Prepare output paths
    os.makedirs(output_dir, exist_ok=True)
    matching_out_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_out_path = os.path.join(output_dir, 'candidate_pairs.tsv')

    print(f"\n6. Processing Test S1 in Batches of {batch_size:,} rows...", flush=True)
    total_s1 = len(test_s1)

    # Initialize TSV output files with headers
    with open(matching_out_path, 'w', encoding='utf-8') as f_match, \
         open(candidate_out_path, 'w', encoding='utf-8') as f_cand:
        
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.flush()
        f_cand.flush()

        for start_idx in range(0, total_s1, batch_size):
            end_idx = min(start_idx + batch_size, total_s1)
            batch_s1 = test_s1.iloc[start_idx:end_idx].copy()

            b_names = dict(zip(batch_s1['entity_id'].values, batch_s1['norm_name'].values))
            b_addrs = dict(zip(batch_s1['entity_id'].values, batch_s1['norm_address'].values))
            b_ctries = dict(zip(batch_s1['entity_id'].values, batch_s1['country'].values))

            # Retrieve candidates for batch
            batch_cands = generate_candidates_for_s1(batch_s1, c_index, top_k=top_k)

            # Build pairwise candidate list
            cand_pairs = []
            pair_index = []
            for s1_id, cands in batch_cands.items():
                for rank, cand_id in enumerate(cands, 1):
                    cand_pairs.append((s1_id, cand_id, rank))
                    pair_index.append((s1_id, cand_id))

            # Extract features and predict
            if len(cand_pairs) > 0:
                X_batch, _ = extract_pairwise_features(b_names, b_addrs, b_ctries, target_names, target_addrs, target_ctries, cand_pairs)
                probs = model.predict(X_batch)
            else:
                probs = np.array([])

            batch_matches = defaultdict(list)
            for idx, (s1_id, cand_id) in enumerate(pair_index):
                if probs[idx] >= threshold:
                    batch_matches[s1_id].append(cand_id)

            # Write batch results to TSV
            for s1_id in batch_s1['entity_id'].values:
                c_list = batch_cands.get(s1_id, [])
                c_str = ",".join(c_list) if c_list else ""
                f_cand.write(f"{s1_id}\t{c_str}\n")

                m_list = batch_matches.get(s1_id, [])
                m_str = ",".join(m_list) if m_list else ""
                f_match.write(f"{s1_id}\t{m_str}\n")

            f_match.flush()
            f_cand.flush()
            print(f"Processed test entities {end_idx:,}/{total_s1:,}...", flush=True)

    print("\n" + "=" * 60)
    print("TEST INFERENCE COMPLETED SUCCESSFULLY!")
    print(f"Saved {matching_out_path}")
    print(f"Saved {candidate_out_path}")
    print("=" * 60)

    return matching_out_path, candidate_out_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', default='dataset/test')
    parser.add_argument('--model-path', default='model/lgbm_model.pkl')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--sample-size', type=int, default=None)
    args = parser.parse_args()

    run_inference(test_dir=args.test_dir, model_path=args.model_path, output_dir=args.output_dir, sample_size=args.sample_size)
