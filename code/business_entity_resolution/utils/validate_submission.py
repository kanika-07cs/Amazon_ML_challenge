import argparse
import os
import sys
import pandas as pd

def validate(matching_file, candidate_file, test_dir):
    print("=" * 60)
    print("RUNNING SUBMISSION VALIDATION")
    print("=" * 60)

    # 1. Load test S1, S2, S3 IDs
    test_s1_path = os.path.join(test_dir, 'test_source1.tsv')
    test_s2_path = os.path.join(test_dir, 'test_source2.tsv')
    test_s3_path = os.path.join(test_dir, 'test_source3.tsv')

    if not os.path.exists(test_s1_path):
        print(f"ERROR: {test_s1_path} not found.")
        sys.exit(1)

    print("Loading test entity IDs...")
    s1_df = pd.read_csv(test_s1_path, sep='\t', usecols=['entity_id'], dtype=str)
    s2_df = pd.read_csv(test_s2_path, sep='\t', usecols=['entity_id'], dtype=str)
    s3_df = pd.read_csv(test_s3_path, sep='\t', usecols=['entity_id'], dtype=str)

    s1_ids = set(s1_df['entity_id'].dropna())
    valid_target_ids = set(s2_df['entity_id'].dropna()).union(set(s3_df['entity_id'].dropna()))

    print(f"Total Test S1 entities expected: {len(s1_ids):,}")
    print(f"Total Valid Target entities (S2 + S3): {len(valid_target_ids):,}")

    # 2. Check files exist
    if not os.path.exists(matching_file):
        print(f"ERROR: Matching output file not found at {matching_file}")
        sys.exit(1)
    if not os.path.exists(candidate_file):
        print(f"ERROR: Candidate output file not found at {candidate_file}")
        sys.exit(1)

    # 3. Load matching results
    print("\nValidating matching_results.tsv...")
    match_df = pd.read_csv(matching_file, sep='\t', dtype=str)
    req_cols_match = ['source1_entity_id', 'matched_entity_ids']
    for col in req_cols_match:
        if col not in match_df.columns:
            print(f"ERROR: Missing column '{col}' in {matching_file}")
            sys.exit(1)

    if len(match_df) != len(s1_ids):
        print(f"ERROR: matching_results.tsv has {len(match_df):,} rows, expected {len(s1_ids):,}")
        sys.exit(1)

    match_s1_set = set(match_df['source1_entity_id'])
    if match_s1_set != s1_ids:
        print(f"ERROR: matching_results.tsv S1 IDs do not match test S1 IDs exactly.")
        sys.exit(1)

    # 4. Load candidate pairs
    print("Validating candidate_pairs.tsv...")
    cand_df = pd.read_csv(candidate_file, sep='\t', dtype=str)
    req_cols_cand = ['source1_entity_id', 'candidate_entity_ids']
    for col in req_cols_cand:
        if col not in cand_df.columns:
            print(f"ERROR: Missing column '{col}' in {candidate_file}")
            sys.exit(1)

    if len(cand_df) != len(s1_ids):
        print(f"ERROR: candidate_pairs.tsv has {len(cand_df):,} rows, expected {len(s1_ids):,}")
        sys.exit(1)

    cand_s1_set = set(cand_df['source1_entity_id'])
    if cand_s1_set != s1_ids:
        print(f"ERROR: candidate_pairs.tsv S1 IDs do not match test S1 IDs exactly.")
        sys.exit(1)

    # 5. Row-by-row deep checks
    print("Performing deep consistency checks on candidates and matches...")
    cand_dict = {}
    for _, row in cand_df.iterrows():
        s1 = row['source1_entity_id']
        c_str = str(row['candidate_entity_ids']) if pd.notnull(row['candidate_entity_ids']) else ''
        c_list = [c.strip() for c in c_str.split(',') if c.strip()]
        if len(c_list) != len(set(c_list)):
            print(f"ERROR: Duplicate candidate IDs for S1 entity {s1}")
            sys.exit(1)
        for c in c_list:
            if c not in valid_target_ids:
                print(f"ERROR: Candidate ID {c} for {s1} is not a valid S2 or S3 test entity ID.")
                sys.exit(1)
        cand_dict[s1] = set(c_list)

    total_matches = 0
    matched_singletons = 0
    for _, row in match_df.iterrows():
        s1 = row['source1_entity_id']
        m_str = str(row['matched_entity_ids']) if pd.notnull(row['matched_entity_ids']) else ''
        m_list = [m.strip() for m in m_str.split(',') if m.strip()]
        if len(m_list) == 0:
            matched_singletons += 1
        else:
            total_matches += len(m_list)

        if len(m_list) != len(set(m_list)):
            print(f"ERROR: Duplicate matched IDs for S1 entity {s1}")
            sys.exit(1)

        c_set = cand_dict.get(s1, set())
        for m in m_list:
            if m not in valid_target_ids:
                print(f"ERROR: Matched ID {m} for {s1} is not a valid S2/S3 test entity ID.")
                sys.exit(1)
            if m not in c_set:
                print(f"ERROR: Matched ID {m} for {s1} is NOT present in candidate_pairs.tsv!")
                sys.exit(1)

    print("\n" + "=" * 60)
    print("VALIDATION PASSED SUCCESSFULLY! ALL CHECKS OK.")
    print("=" * 60)
    print(f"Total S1 entities validated: {len(s1_ids):,}")
    print(f"Predicted Singletons (0 matches): {matched_singletons:,}")
    print(f"Predicted Matched S1 Entities: {len(s1_ids) - matched_singletons:,}")
    print(f"Total Individual Matched Records: {total_matches:,}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matching', default='output/matching_results.tsv', help="Path to matching_results.tsv")
    parser.add_argument('--candidate', default='output/candidate_pairs.tsv', help="Path to candidate_pairs.tsv")
    parser.add_argument('--test-dir', default='dataset/test', help="Path to test dataset directory")
    args = parser.parse_args()

    validate(args.matching, args.candidate, args.test_dir)
