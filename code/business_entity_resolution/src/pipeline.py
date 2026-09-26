import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import train_and_tune_model
from infer import run_inference

def main():
    parser = argparse.ArgumentParser(description="Run complete Business Entity Resolution pipeline.")
    parser.add_argument('--dataset-dir', default='dataset', help="Path to dataset directory containing train/ and test/")
    parser.add_argument('--sample-size', type=int, default=10000, help="Number of train S1 entities to use for training/tuning sample")
    parser.add_argument('--model-path', default='model/lgbm_model.pkl', help="Path to save/load trained model artifact")
    parser.add_argument('--output-dir', default='output', help="Directory to store output TSV files")
    parser.add_argument('--top-k', type=int, default=25, help="Top K candidates to generate per entity")
    args = parser.parse_args()

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    # Step 1: Train model & tune threshold
    train_and_tune_model(
        dataset_dir=args.dataset_dir,
        sample_size=args.sample_size,
        model_save_path=args.model_path
    )

    # Step 2: Test Inference & Output Generation
    test_dir = os.path.join(args.dataset_dir, 'test')
    matching_file, candidate_file = run_inference(
        test_dir=test_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        top_k=args.top_k
    )

    # Step 3: Run Official Submission Validator
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING SUBMISSION VALIDATION")
    print("=" * 70)
    validator_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'validate_submission.py')
    
    val_cmd = f"python \"{validator_path}\" --matching \"{matching_file}\" --candidate \"{candidate_file}\" --test-dir \"{test_dir}\""
    print(f"Executing: {val_cmd}")
    ret_code = os.system(val_cmd)
    
    if ret_code == 0:
        print("\nPIPELINE COMPLETED SUCCESSFULLY! Output passed 100% submission validation.")
    else:
        print(f"\nPipeline finished with validation exit code {ret_code}.")

if __name__ == '__main__':
    main()
