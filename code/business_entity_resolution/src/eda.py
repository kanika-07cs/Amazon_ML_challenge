import pandas as pd
import numpy as np
import os
import sys

def explore_dataset(dataset_dir):
    print("=" * 60, flush=True)
    print("DATASET EXPLORATION REPORT", flush=True)
    print("=" * 60, flush=True)
    
    # 1. Load train files
    print("Loading train files...", flush=True)
    train_s1 = pd.read_csv(os.path.join(dataset_dir, 'train', 'train_source1.tsv'), sep='\t', dtype=str)
    print(f"Loaded Train S1: {len(train_s1):,} rows", flush=True)
    train_s2 = pd.read_csv(os.path.join(dataset_dir, 'train', 'train_source2.tsv'), sep='\t', dtype=str)
    print(f"Loaded Train S2: {len(train_s2):,} rows", flush=True)
    train_s3 = pd.read_csv(os.path.join(dataset_dir, 'train', 'train_source3.tsv'), sep='\t', dtype=str)
    print(f"Loaded Train S3: {len(train_s3):,} rows", flush=True)
    train_gt = pd.read_csv(os.path.join(dataset_dir, 'train', 'train_ground_truth.tsv'), sep='\t', dtype=str)
    print(f"Loaded Train Ground Truth: {len(train_gt):,} rows", flush=True)

    # 2. Load test files
    print("\nLoading test files...", flush=True)
    test_s1 = pd.read_csv(os.path.join(dataset_dir, 'test', 'test_source1.tsv'), sep='\t', dtype=str)
    print(f"Loaded Test S1: {len(test_s1):,} rows", flush=True)
    test_s2 = pd.read_csv(os.path.join(dataset_dir, 'test', 'test_source2.tsv'), sep='\t', dtype=str)
    print(f"Loaded Test S2: {len(test_s2):,} rows", flush=True)
    test_s3 = pd.read_csv(os.path.join(dataset_dir, 'test', 'test_source3.tsv'), sep='\t', dtype=str)
    print(f"Loaded Test S3: {len(test_s3):,} rows", flush=True)

    print("\n--- MISSING VALUES IN TRAIN ---", flush=True)
    for name, df in [('Train S1', train_s1), ('Train S2', train_s2), ('Train S3', train_s3)]:
        print(f"\n{name} missing counts:", flush=True)
        print(df.isnull().sum(), flush=True)

    print("\n--- MISSING VALUES IN TEST ---", flush=True)
    for name, df in [('Test S1', test_s1), ('Test S2', test_s2), ('Test S3', test_s3)]:
        print(f"\n{name} missing counts:", flush=True)
        print(df.isnull().sum(), flush=True)

    print("\n--- GROUND TRUTH ANALYSIS ---", flush=True)
    # Clean ground truth matched_entity_ids
    train_gt['matched_list'] = train_gt['matched_entity_ids'].fillna('').apply(
        lambda x: [i.strip() for i in str(x).split(',') if i.strip()]
    )
    train_gt['match_count'] = train_gt['matched_list'].apply(len)

    print(f"Total Source 1 entities in ground truth: {len(train_gt):,}", flush=True)
    print(f"Entities with 0 matches (singletons): {(train_gt['match_count'] == 0).sum():,} ({((train_gt['match_count'] == 0).mean() * 100):.2f}%)", flush=True)
    print(f"Entities with >0 matches: {(train_gt['match_count'] > 0).sum():,} ({((train_gt['match_count'] > 0).mean() * 100):.2f}%)", flush=True)
    
    print("\nMatch count distribution:", flush=True)
    print(train_gt['match_count'].value_counts().sort_index().head(15), flush=True)
    print(f"Max matches for a single entity: {train_gt['match_count'].max()}", flush=True)
    print(f"Average matches per matched S1 entity: {train_gt[train_gt['match_count'] > 0]['match_count'].mean():.2f}", flush=True)

    # Count S2 vs S3 matches
    all_matched_ids = [m for sublist in train_gt['matched_list'] for m in sublist]
    s2_matches = sum(1 for m in all_matched_ids if m.startswith('S2-'))
    s3_matches = sum(1 for m in all_matched_ids if m.startswith('S3-'))
    other_matches = len(all_matched_ids) - s2_matches - s3_matches

    print(f"\nTotal matched records across all S1: {len(all_matched_ids):,}", flush=True)
    print(f"Matches from Source 2: {s2_matches:,} ({s2_matches/len(all_matched_ids)*100:.2f}%)", flush=True)
    print(f"Matches from Source 3: {s3_matches:,} ({s3_matches/len(all_matched_ids)*100:.2f}%)", flush=True)
    if other_matches > 0:
        print(f"Matches from unknown prefix: {other_matches:,}", flush=True)

    # Country breakdown
    print("\n--- COUNTRY DISTRIBUTION (TRAIN S1 Top 10) ---", flush=True)
    print(train_s1['country'].value_counts(dropna=False).head(10), flush=True)

    print("\n--- COUNTRY DISTRIBUTION (TEST S1 Top 10) ---", flush=True)
    print(test_s1['country'].value_counts(dropna=False).head(10), flush=True)

if __name__ == '__main__':
    dataset_path = 'dataset'
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    explore_dataset(dataset_path)
