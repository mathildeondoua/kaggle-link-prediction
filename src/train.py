"""
train.py — Model training and cross-validation for link prediction.

Trains multiple classical ML models, evaluates with StratifiedKFold cross-validation
on AUC-ROC, and saves the best model.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_node_features, load_train, build_graph
from feature_extractor import extract_all_features

# Output directory for saved models
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def get_models():
    """
    Define the candidate models to evaluate.
    
    Returns:
        dict of model_name -> (model_instance, needs_scaling)
    """
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42),
            True  # LR benefits from scaling
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=500, max_depth=12, min_samples_leaf=5,
                n_jobs=-1, random_state=42
            ),
            False
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="auc", random_state=42, verbosity=0
            ),
            False
        ),
        "LightGBM": (
            lgb.LGBMClassifier(
                n_estimators=500, max_depth=8, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            ),
            False
        ),
    }
    return models


def cross_validate_model(model, X, y, n_splits=5, needs_scaling=False):
    """
    Evaluate a model using StratifiedKFold cross-validation.
    
    Args:
        model: sklearn-compatible classifier
        X: feature matrix (DataFrame or np.array)
        y: labels (np.array)
        n_splits: number of CV folds
        needs_scaling: whether to apply StandardScaler
    
    Returns:
        dict with 'auc_scores' (list), 'mean_auc', 'std_auc'
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if needs_scaling:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
        
        model_clone = _clone_model(model)
        model_clone.fit(X_train, y_train)
        
        # Predict probabilities for the positive class
        y_proba = model_clone.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        auc_scores.append(auc)
        
        print(f"    Fold {fold_idx + 1}/{n_splits}: AUC = {auc:.4f}")
    
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    return {"auc_scores": auc_scores, "mean_auc": mean_auc, "std_auc": std_auc}


def _clone_model(model):
    """Clone a model with the same hyperparameters."""
    from sklearn.base import clone
    return clone(model)


def train_and_evaluate_all(X, y):
    """
    Train and cross-validate all candidate models.
    
    Args:
        X: feature DataFrame
        y: label array
    
    Returns:
        results: dict of model_name -> cv_results
        best_model_name: name of the best model
    """
    models = get_models()
    results = {}
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS (5-fold, metric=AUC-ROC)")
    print(f"{'='*60}")
    
    for name, (model, needs_scaling) in models.items():
        print(f"\n--- {name} ---")
        start = time.time()
        cv_result = cross_validate_model(model, X, y, n_splits=5, needs_scaling=needs_scaling)
        elapsed = time.time() - start
        
        results[name] = cv_result
        print(f"  => Mean AUC: {cv_result['mean_auc']:.4f} (+/- {cv_result['std_auc']:.4f}) "
              f"[{elapsed:.1f}s]")
    
    # Find best model
    best_name = max(results, key=lambda k: results[k]["mean_auc"])
    best_auc = results[best_name]["mean_auc"]
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name} (Mean AUC = {best_auc:.4f})")
    print(f"{'='*60}")
    
    return results, best_name


def train_final_model(X, y, model_name):
    """
    Train the selected model on the full training data and save it.
    
    Args:
        X: full feature DataFrame
        y: full label array
        model_name: name of the model to train
    
    Returns:
        trained model, scaler (or None)
    """
    models = get_models()
    model, needs_scaling = models[model_name]
    
    scaler = None
    if needs_scaling:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        model.fit(X_scaled, y)
    else:
        model.fit(X, y)
    
    # Save model and scaler
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump({"model": model, "scaler": scaler, "model_name": model_name}, model_path)
    print(f"\n[train] Model saved to {model_path}")
    
    return model, scaler


def print_feature_importances(model, feature_names, model_name, top_n=20):
    """
    Print and return feature importances for tree-based models, 
    or coefficients for linear models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        importance_type = "Absolute Coefficient"
    else:
        print("[train] Model does not expose feature importances.")
        return None
    
    fi_df = pd.DataFrame({
        "feature": feature_names,
        importance_type: importances
    }).sort_values(importance_type, ascending=False)
    
    print(f"\n--- Top {top_n} Features ({model_name}) ---")
    print(fi_df.head(top_n).to_string(index=False))
    
    return fi_df


def main():
    """Full training pipeline."""
    print("=" * 60)
    print("LINK PREDICTION — TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    print("\n[Step 1] Loading data...")
    node_ids, features_matrix, node_id_to_idx = load_node_features()
    train_df = load_train()
    G = build_graph(train_df, node_ids)
    
    # 2. Extract features
    print("\n[Step 2] Extracting features...")
    X = extract_all_features(G, features_matrix, node_id_to_idx, train_df, is_training=True)
    y = train_df["label"].values
    
    # Sanity checks
    assert X.shape[0] == len(y), f"Mismatch: {X.shape[0]} features vs {len(y)} labels"
    assert X.isna().sum().sum() == 0, "NaN values found in features!"
    assert np.isinf(X.values).sum() == 0, "Inf values found in features!"
    print(f"\n[Sanity] Feature matrix: {X.shape}, Label balance: {pd.Series(y).value_counts().to_dict()}")
    
    # 3. Cross-validate all models
    print("\n[Step 3] Cross-validating models...")
    results, best_model_name = train_and_evaluate_all(X, y)
    
    # 4. Train final model on full data
    print(f"\n[Step 4] Training final {best_model_name} on full data...")
    model, scaler = train_final_model(X, y, best_model_name)
    
    # 5. Feature importances
    print("\n[Step 5] Feature importances...")
    fi_df = print_feature_importances(model, X.columns.tolist(), best_model_name)
    
    # 6. Save feature names for reproducibility
    feature_names_path = os.path.join(MODEL_DIR, "feature_names.txt")
    with open(feature_names_path, "w") as f:
        f.write("\n".join(X.columns.tolist()))
    print(f"[train] Feature names saved to {feature_names_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Best model: {best_model_name}")
    print(f"  CV AUC: {results[best_model_name]['mean_auc']:.4f} "
          f"(+/- {results[best_model_name]['std_auc']:.4f})")
    print(f"  Features used: {X.shape[1]}")
    print(f"{'='*60}")
    
    return results, best_model_name


if __name__ == "__main__":
    main()
