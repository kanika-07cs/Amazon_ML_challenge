import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_dataframe
from blocking import build_candidate_index, generate_candidates_for_s1, evaluate_blocking_recall
from feature_engineering import extract_pairwise_features
from evaluate import evaluate_macro_f05

def train_and_tune_model(dataset_dir='dataset', sample_size=10000, model_save_path='model/lgbm_model.pkl'):
    print("=" * 60)
    print("STAGE 6 & 7: MODEL TRAINING & THRESHOLD TUNING")
    print("=" * 60)
    
    # 1. Load Ground Truth
    print("\n1. Loading Ground Truth...", flush=True)
    gt_df = pd.read_csv(os.path.join(dataset_dir, 'train', 'train_ground_truth.tsv'), sep='\t', nrows=sample_size, dtype=str)
    gt_df['matched_list'] = gt_df['matched_entity_ids'].fillna('').apply(
        lambda x: [i.strip() for i in str(x).split(',') if i.strip()]
    )

    # Split ground truth into Train (80%) and Validation (20%) S1 entities
    train_gt, val_gt = train_test_split(gt_df, test_size=0.20, random_state=42)
    print(f"Train S1 Entities: {len(train_gt):,}")
    print(f"Val S1 Entities:   {len(val_gt):,}")

    # 2. Load and Preprocess S1, S2, S3
    print("\n2. Loading Data Sources...", flush=True)
    s1_all = pd.read_csv(os.path.join(dataset_dir, 'train', 'train_source1.tsv'), sep='\t', dtype=str)
    s1_all = preprocess_dataframe(s1_all)

    train_s1 = s1_all[s1_all['entity_id'].isin(set(train_gt['source1_entity_id']))].copy()
    val_s1 = s1_all[s1_all['entity_id'].isin(set(val_gt['source1_entity_id']))].copy()

    s2_df = pd.read_csv(os.path.join(dataset_dir, 'train', 'train_source2.tsv'), sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    s2_df = preprocess_dataframe(s2_df)

    s3_df = pd.read_csv(os.path.join(dataset_dir, 'train', 'train_source3.tsv'), sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'], dtype=str)
    s3_df = preprocess_dataframe(s3_df)

    # 3. Build Blocking Index
    print("\n3. Building Candidate Blocking Index...", flush=True)
    c_index = build_candidate_index(s2_df, s3_df, max_posting_list=3000)

    # 4. Generate Candidates for Train and Val
    print("\n4. Generating Candidates for Training Set...", flush=True)
    train_cands = generate_candidates_for_s1(train_s1, c_index, top_k=25)
    print("Train Candidate Recall:")
    evaluate_blocking_recall(train_cands, train_gt)

    print("\n5. Generating Candidates for Validation Set...", flush=True)
    val_cands = generate_candidates_for_s1(val_s1, c_index, top_k=25)
    print("Validation Candidate Recall:")
    evaluate_blocking_recall(val_cands, val_gt)

    # 5. Build Sub-Second Record Lookups using zip dicts
    print("\n6. Preparing Record Lookups...", flush=True)
    t0 = time.time()
    s1_names = dict(zip(s1_all['entity_id'].values, s1_all['norm_name'].values))
    s1_addrs = dict(zip(s1_all['entity_id'].values, s1_all['norm_address'].values))
    s1_ctries = dict(zip(s1_all['entity_id'].values, s1_all['country'].values))

    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    t_eids = target_df['entity_id'].values
    target_names = dict(zip(t_eids, target_df['norm_name'].values))
    target_addrs = dict(zip(t_eids, target_df['norm_address'].values))
    target_ctries = dict(zip(t_eids, target_df['country'].values))
    del s2_df, s3_df, target_df
    print(f"Record lookups created in {time.time() - t0:.2f}s", flush=True)

    # 6. Build Labeled Candidate Pairs for Training
    print("\n7. Building Training Pair Dataset & Feature Engineering...", flush=True)
    train_gt_dict = train_gt.set_index('source1_entity_id')['matched_list'].to_dict()
    
    train_pairs = []
    train_labels = []
    for s1_id, cands in train_cands.items():
        true_set = set(train_gt_dict.get(s1_id, []))
        for rank, cand_id in enumerate(cands, 1):
            train_pairs.append((s1_id, cand_id, rank))
            train_labels.append(1 if cand_id in true_set else 0)

    train_labels = np.array(train_labels, dtype=np.int32)
    print(f"Total Train Pairs: {len(train_pairs):,} (Positives: {np.sum(train_labels):,}, Negatives: {len(train_labels) - np.sum(train_labels):,})")

    X_train, feature_names = extract_pairwise_features(s1_names, s1_addrs, s1_ctries, target_names, target_addrs, target_ctries, train_pairs)

    # 7. Train LightGBM Classifier
    print("\n8. Training LightGBM Pairwise Classifier...", flush=True)
    train_data = lgb.Dataset(X_train, label=train_labels, feature_name=feature_names)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    model = lgb.train(params, train_data, num_boost_round=200)

    # 8. Build Feature Vectors for Validation Candidate Pairs
    print("\n9. Extracting Features for Validation Set...", flush=True)
    val_pairs = []
    val_pair_index = []  # keeps track of (s1_id, cand_id)
    for s1_id, cands in val_cands.items():
        for rank, cand_id in enumerate(cands, 1):
            val_pairs.append((s1_id, cand_id, rank))
            val_pair_index.append((s1_id, cand_id))

    X_val, _ = extract_pairwise_features(s1_names, s1_addrs, s1_ctries, target_names, target_addrs, target_ctries, val_pairs)
    val_probs = model.predict(X_val)

    # 9. Threshold Tuning on Validation Set for Macro F0.5
    print("\n10. Tuning Classification Threshold on Validation Set...", flush=True)
    best_threshold = 0.5
    best_f05 = -1.0

    thresholds = np.linspace(0.1, 0.9, 17)
    for tau in thresholds:
        pred_dict = defaultdict(list)
        for idx, (s1_id, cand_id) in enumerate(val_pair_index):
            if val_probs[idx] >= tau:
                pred_dict[s1_id].append(cand_id)

        metrics = evaluate_macro_f05(pred_dict, val_gt)
        f05 = metrics['macro_f05']
        print(f"Threshold tau = {tau:.2f} -> Macro F0.5 = {f05:.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = tau

    print("\n" + "=" * 60)
    print(f"OPTIMAL THRESHOLD FOUND: tau* = {best_threshold:.2f} (Macro F0.5 = {best_f05:.4f})")
    print("=" * 60)

    # Save model and metadata
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    save_obj = {
        'model': model,
        'threshold': best_threshold,
        'feature_names': feature_names
    }
    joblib.dump(save_obj, model_save_path)
    print(f"Saved trained model artifacts to {model_save_path}")

if __name__ == '__main__':
    train_and_tune_model(dataset_dir='dataset', sample_size=10000, model_save_path='model/lgbm_model.pkl')
