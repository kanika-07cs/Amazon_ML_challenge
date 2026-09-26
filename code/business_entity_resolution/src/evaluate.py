import numpy as np
import pandas as pd

def compute_entity_f05(gt_set, pred_set):
    """
    Compute F0.5 score for a single Source 1 entity.
    Includes singleton entities.
    """
    # Ground truth is singleton (no true matches)
    if len(gt_set) == 0:
        if len(pred_set) == 0:
            return 1.0  # Correct singleton prediction
        else:
            return 0.0  # False positive prediction on a singleton
            
    # Ground truth has matches
    if len(pred_set) == 0:
        return 0.0  # False negative (missed all true matches)
        
    intersection = len(gt_set.intersection(pred_set))
    if intersection == 0:
        return 0.0
        
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(gt_set))
    
    beta_sq = 0.25  # 0.5^2
    f05 = (1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    return f05

def evaluate_macro_f05(predictions_dict, ground_truth_df):
    """
    Compute Macro-Averaged F0.5 score across all Source 1 entities in ground truth.
    
    Parameters:
    - predictions_dict: dict {s1_id: set or list of predicted matched entity IDs}
    - ground_truth_df: DataFrame with columns 'source1_entity_id', 'matched_entity_ids'
    
    Returns:
    - Dict with macro_f05, mean_precision, mean_recall, singleton_accuracy
    """
    f05_scores = []
    singleton_count = 0
    correct_singletons = 0
    matched_entity_count = 0
    
    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        m_str = str(row['matched_entity_ids']) if pd.notnull(row['matched_entity_ids']) else ''
        gt_set = set([m.strip() for m in m_str.split(',') if m.strip()])
        
        pred_list = predictions_dict.get(s1_id, [])
        pred_set = set(pred_list)
        
        score = compute_entity_f05(gt_set, pred_set)
        f05_scores.append(score)
        
        if len(gt_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                correct_singletons += 1
        else:
            matched_entity_count += 1

    macro_f05 = float(np.mean(f05_scores))
    singleton_acc = (correct_singletons / float(singleton_count) * 100.0) if singleton_count > 0 else 0.0
    
    print("\n" + "=" * 50)
    print("CHALLENGE EVALUATION METRIC RESULTS")
    print("=" * 50)
    print(f"Total Evaluated S1 Entities: {len(ground_truth_df):,}")
    print(f"True Singleton Entities: {singleton_count:,}")
    print(f"Singleton Accuracy: {singleton_acc:.2f}% ({correct_singletons:,}/{singleton_count:,})")
    print(f"True Matched Entities: {matched_entity_count:,}")
    print(f"Macro-Averaged F0.5 Score: {macro_f05:.4f}")
    print("=" * 50)

    return {
        'macro_f05': macro_f05,
        'singleton_accuracy': singleton_acc,
        'total_s1': len(ground_truth_df)
    }
