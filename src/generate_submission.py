"""
generate_submission.py — Generate Kaggle submission file for link prediction.

Loads the trained model, extracts features for test pairs,
and outputs a CSV with probabilities in the required format.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_node_features, load_train, load_test, build_graph
from feature_extractor import extract_all_features

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submissions")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_submission(output_filename="submission.csv"):
    """
    Generate the Kaggle submission file.
    
    Steps:
        1. Load the trained model
        2. Load data and build graph
        3. Extract features for test pairs
        4. Predict probabilities
        5. Write CSV in the format: ID, Predicted
    """
    print("=" * 60)
    print("GENERATING SUBMISSION")
    print("=" * 60)
    
    # 1. Load model
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}. Run train.py first.")
    
    saved = joblib.load(model_path)
    model = saved["model"]
    scaler = saved["scaler"]
    model_name = saved["model_name"]
    print(f"[submission] Loaded model: {model_name}")
    
    # 2. Load data
    node_ids, features_matrix, node_id_to_idx = load_node_features()
    train_df = load_train()
    test_df = load_test()
    G = build_graph(train_df, node_ids)
    
    # 3. Extract features for test pairs
    print("\n[submission] Extracting features for test pairs...")
    X_test = extract_all_features(G, features_matrix, node_id_to_idx, test_df)
    
    # Sanity checks
    assert X_test.shape[0] == len(test_df), f"Mismatch: {X_test.shape[0]} vs {len(test_df)}"
    assert X_test.isna().sum().sum() == 0, "NaN in test features!"
    
    # 4. Apply scaling if needed
    if scaler is not None:
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # 5. Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Sanity checks on predictions
    print(f"\n[submission] Prediction statistics:")
    print(f"  Min probability: {y_proba.min():.4f}")
    print(f"  Max probability: {y_proba.max():.4f}")
    print(f"  Mean probability: {y_proba.mean():.4f}")
    print(f"  Std probability: {y_proba.std():.4f}")
    print(f"  Predicted positive (>0.5): {(y_proba > 0.5).sum()} / {len(y_proba)}")
    
    # 6. Write submission CSV
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    submission = pd.DataFrame({
        "ID": np.arange(len(y_proba)),
        "Predicted": y_proba
    })
    submission.to_csv(output_path, index=False)
    
    print(f"\n[submission] Saved to {output_path}")
    print(f"  Rows: {len(submission)} (expected: {len(test_df)})")
    print(f"  Columns: {list(submission.columns)}")
    
    # Final verification
    assert len(submission) == 3498, f"Expected 3498 rows, got {len(submission)}"
    assert all(0 <= p <= 1 for p in y_proba), "Probabilities out of [0, 1] range!"
    
    print(f"\n{'='*60}")
    print(f"SUBMISSION GENERATED SUCCESSFULLY")
    print(f"{'='*60}")
    
    return submission


if __name__ == "__main__":
    generate_submission()
