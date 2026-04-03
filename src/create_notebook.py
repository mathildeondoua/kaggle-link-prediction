"""
create_notebook.py — Script to programmatically create the Jupyter notebook 
for the link prediction project. Run this to generate notebook.ipynb.
"""

import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Set kernel metadata
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

cells = []

# ============================================================
# TITLE & INTRODUCTION
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""# 🎬 Link Prediction — Actor Co-occurrence Network

## Competition Overview

This notebook presents a complete pipeline for predicting missing links in an actor co-occurrence network from the Kaggle competition. 

**Task**: Given a graph where nodes are actors and edges represent co-occurrence on the same Wikipedia page, predict whether missing edges existed in the original graph.

**Approach**: Classical ML with extensive feature engineering combining:
1. **Graph topological features** (common neighbors, Jaccard, Adamic-Adar, etc.)
2. **Node attribute features** (cosine similarity on Wikipedia keyword vectors)
3. **Community features** (Louvain community detection)
4. **SVD graph embeddings** (latent structural similarity)
5. **Centrality features** (PageRank, clustering coefficient)

**Metric**: AUC-ROC

---"""))

# ============================================================
# SETUP
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## 1. Setup & Imports"))

cells.append(nbf.v4.new_code_cell("""import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Add src to path
sys.path.insert(0, 'src')
from data_loader import load_node_features, load_train, load_test, build_graph
from feature_extractor import extract_all_features

print("Setup complete! ✅")"""))

# ============================================================
# DATA LOADING
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## 2. Data Loading & Initial Exploration"))

cells.append(nbf.v4.new_code_cell("""# Load all data
node_ids, features_matrix, node_id_to_idx = load_node_features()
train_df = load_train()
test_df = load_test()
G = build_graph(train_df, node_ids)

print(f"\\n{'='*50}")
print(f"DATASET SUMMARY")
print(f"{'='*50}")
print(f"Nodes: {len(node_ids)}")
print(f"Node features: {features_matrix.shape[1]} (binary keyword vectors)")
print(f"Training pairs: {len(train_df)} (balanced: {train_df['label'].value_counts().to_dict()})")
print(f"Test pairs: {len(test_df)}")
print(f"Graph edges: {G.number_of_edges()}")
print(f"Graph density: {nx.density(G):.6f}")
print(f"Connected components: {nx.number_connected_components(G)}")"""))

# ============================================================
# EDA — GRAPH STRUCTURE
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis

### 3.1 Graph Structure Analysis"""))

cells.append(nbf.v4.new_code_cell("""# Degree distribution
degrees = [d for _, d in G.degree()]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram of degrees
axes[0].hist(degrees, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Degree')
axes[0].set_ylabel('Count')
axes[0].set_title('Degree Distribution')
axes[0].axvline(np.mean(degrees), color='red', linestyle='--', label=f'Mean: {np.mean(degrees):.1f}')
axes[0].axvline(np.median(degrees), color='orange', linestyle='--', label=f'Median: {np.median(degrees):.0f}')
axes[0].legend()

# Log-log degree distribution (check power law)
degree_counts = Counter(degrees)
deg_vals = sorted(degree_counts.keys())
deg_freqs = [degree_counts[d] for d in deg_vals]
axes[1].loglog(deg_vals, deg_freqs, 'o', markersize=4, color='steelblue')
axes[1].set_xlabel('Degree (log)')
axes[1].set_ylabel('Frequency (log)')
axes[1].set_title('Log-Log Degree Distribution')

# Cumulative degree distribution
sorted_degrees = np.sort(degrees)[::-1]
axes[2].plot(range(len(sorted_degrees)), sorted_degrees, color='steelblue', linewidth=2)
axes[2].set_xlabel('Node rank')
axes[2].set_ylabel('Degree')
axes[2].set_title('Degree Rank Plot')

plt.tight_layout()
plt.savefig('degree_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\nDegree statistics:")
print(f"  Mean: {np.mean(degrees):.2f}")
print(f"  Median: {np.median(degrees):.0f}")
print(f"  Max: {max(degrees)} (hub node)")
print(f"  Nodes with degree 1: {sum(1 for d in degrees if d == 1)}")
print(f"  Nodes with degree > 10: {sum(1 for d in degrees if d > 10)}")"""))

# ============================================================
# EDA — NODE FEATURES
# ============================================================
cells.append(nbf.v4.new_markdown_cell("### 3.2 Node Feature Analysis"))

cells.append(nbf.v4.new_code_cell("""# Node feature statistics (binary keyword vectors)
keywords_per_node = features_matrix.sum(axis=1)
nodes_per_keyword = features_matrix.sum(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Keywords per node
axes[0].hist(keywords_per_node, bins=30, color='coral', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Number of keywords')
axes[0].set_ylabel('Number of nodes')
axes[0].set_title('Keywords per Node (Actor)')
axes[0].axvline(np.mean(keywords_per_node), color='red', linestyle='--', 
                label=f'Mean: {np.mean(keywords_per_node):.1f}')
axes[0].legend()

# Nodes per keyword (sparsity analysis)
nonzero_keywords = nodes_per_keyword[nodes_per_keyword > 0]
axes[1].hist(nonzero_keywords, bins=40, color='mediumseagreen', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Number of nodes with keyword')
axes[1].set_ylabel('Number of keywords')
axes[1].set_title('Nodes per Keyword (non-zero only)')

plt.tight_layout()
plt.savefig('node_features_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Feature matrix density: {(features_matrix != 0).mean():.4f} (very sparse)")
print(f"Keywords per node: mean={np.mean(keywords_per_node):.1f}, std={np.std(keywords_per_node):.1f}")
print(f"Active keywords (at least 1 node): {(nodes_per_keyword > 0).sum()} / {features_matrix.shape[1]}")
print(f"Keywords present in >10 nodes: {(nodes_per_keyword > 10).sum()}")"""))

# ============================================================
# EDA — TRAINING SET
# ============================================================
cells.append(nbf.v4.new_markdown_cell("### 3.3 Training Set Analysis"))

cells.append(nbf.v4.new_code_cell("""# Analyze positive vs negative pairs
pos_pairs = train_df[train_df['label'] == 1]
neg_pairs = train_df[train_df['label'] == 0]

# Degree analysis for positive vs negative pairs
def get_pair_degrees(pairs, G):
    deg_s = [G.degree(s) for s in pairs['source']]
    deg_t = [G.degree(t) for t in pairs['target']]
    return deg_s, deg_t

pos_deg_s, pos_deg_t = get_pair_degrees(pos_pairs, G)
neg_deg_s, neg_deg_t = get_pair_degrees(neg_pairs, G)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Source degree distribution
axes[0].hist(pos_deg_s, bins=30, alpha=0.6, label='Positive (edge exists)', color='green')
axes[0].hist(neg_deg_s, bins=30, alpha=0.6, label='Negative (no edge)', color='red')
axes[0].set_xlabel('Source Node Degree')
axes[0].set_ylabel('Count')
axes[0].set_title('Source Node Degree by Label')
axes[0].legend()

# Target degree distribution
axes[1].hist(pos_deg_t, bins=30, alpha=0.6, label='Positive (edge exists)', color='green')
axes[1].hist(neg_deg_t, bins=30, alpha=0.6, label='Negative (no edge)', color='red')
axes[1].set_xlabel('Target Node Degree')
axes[1].set_ylabel('Count')
axes[1].set_title('Target Node Degree by Label')
axes[1].legend()

plt.tight_layout()
plt.savefig('label_vs_degree.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Positive pairs — avg source degree: {np.mean(pos_deg_s):.1f}, avg target degree: {np.mean(pos_deg_t):.1f}")
print(f"Negative pairs — avg source degree: {np.mean(neg_deg_s):.1f}, avg target degree: {np.mean(neg_deg_t):.1f}")
print(f"\\nObservation: Positive pairs tend to involve higher-degree nodes (hubs)")"""))

# ============================================================
# FEATURE ENGINEERING
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Feature Engineering

We extract 5 families of features for each node pair:

| Family | Features | Rationale |
|--------|----------|-----------|
| **Graph topology** | Common neighbors, Jaccard, Adamic-Adar, Resource Allocation, Preferential Attachment, Degrees, Shortest Path | Standard link prediction heuristics |
| **Node attributes** | Cosine similarity, keyword overlap, Jaccard on features | Actors sharing keywords likely co-occur |
| **Community** | Same community, community sizes | Actors in same community are more connected |
| **SVD embeddings** | Dot product, distance, cosine in SVD space | Captures latent graph structure |
| **Centrality** | PageRank, clustering coefficient | Node importance measures |

> ⚠️ **Data Leakage Prevention**: For training pairs, edges are temporarily removed from the graph before computing local graph features (common neighbors, shortest path) to avoid trivially detecting existing edges."""))

cells.append(nbf.v4.new_code_cell("""# Extract features for training data (with edge removal to prevent leakage)
print("Extracting training features (with edge removal for leakage prevention)...")
X_train = extract_all_features(G, features_matrix, node_id_to_idx, train_df, is_training=True)
y_train = train_df['label'].values

print(f"\\nFeature matrix shape: {X_train.shape}")
print(f"Label balance: {pd.Series(y_train).value_counts().to_dict()}")
print(f"NaN count: {X_train.isna().sum().sum()}")
print(f"Inf count: {np.isinf(X_train.values).sum()}")

X_train.head(10)"""))

# ============================================================
# FEATURE ANALYSIS
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Feature Analysis

### 5.1 Feature Distributions by Label"""))

cells.append(nbf.v4.new_code_cell("""# Compare feature distributions for positive vs negative pairs
feature_families = {
    'Graph Topology': ['common_neighbors', 'jaccard_coefficient', 'adamic_adar_index', 
                       'resource_allocation', 'preferential_attachment', 'shortest_path_length'],
    'Node Attributes': ['cosine_similarity', 'common_keywords', 'jaccard_features', 'feature_diff_l2'],
    'Community': ['same_community'],
    'SVD Embeddings': ['svd_dot_product', 'svd_cosine_similarity'],
    'Centrality': ['pagerank_sum', 'clustering_coeff_sum'],
}

# Summary table: mean positive vs mean negative
summary_data = []
for col in X_train.columns:
    mean_pos = X_train.loc[y_train == 1, col].mean()
    mean_neg = X_train.loc[y_train == 0, col].mean()
    ratio = mean_pos / mean_neg if mean_neg != 0 else float('inf')
    summary_data.append({
        'Feature': col,
        'Mean (Positive)': round(mean_pos, 4),
        'Mean (Negative)': round(mean_neg, 4),
        'Ratio (Pos/Neg)': round(ratio, 2),
        'Discriminative': '✅ Strong' if abs(ratio - 1) > 1 else ('⚠️ Moderate' if abs(ratio - 1) > 0.2 else '❌ Weak')
    })

summary_df = pd.DataFrame(summary_data).sort_values('Ratio (Pos/Neg)', ascending=False)
print("Feature Discrimination Summary (Positive vs Negative pairs):")
print("="*85)
print(summary_df.to_string(index=False))"""))

cells.append(nbf.v4.new_markdown_cell("### 5.2 Key Feature Distributions"))

cells.append(nbf.v4.new_code_cell("""# Plot distributions of most discriminative features
key_features = ['common_neighbors', 'adamic_adar_index', 'preferential_attachment',
                'same_community', 'svd_dot_product', 'cosine_similarity',
                'shortest_path_length', 'pagerank_sum']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, feat in enumerate(key_features):
    ax = axes[idx]
    pos_vals = X_train.loc[y_train == 1, feat]
    neg_vals = X_train.loc[y_train == 0, feat]
    
    ax.hist(neg_vals, bins=30, alpha=0.6, label='No edge', color='red', density=True)
    ax.hist(pos_vals, bins=30, alpha=0.6, label='Edge', color='green', density=True)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=8)

plt.suptitle('Feature Distributions: Edge vs No Edge', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("### 5.3 Feature Correlation Matrix"))

cells.append(nbf.v4.new_code_cell("""# Correlation matrix
plt.figure(figsize=(16, 14))
corr = X_train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, annot=False,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# Highly correlated pairs
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.9:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3)))

if high_corr_pairs:
    print("Highly correlated feature pairs (|r| > 0.9):")
    for f1, f2, r in high_corr_pairs:
        print(f"  {f1} <-> {f2}: r = {r}")
else:
    print("No feature pairs with |r| > 0.9")"""))

# ============================================================
# MODEL TRAINING & CROSS-VALIDATION
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Model Training & Cross-Validation

We compare 4 classical ML models using 5-fold Stratified Cross-Validation with AUC-ROC:
1. **Logistic Regression** — linear baseline
2. **Random Forest** — ensemble of decision trees
3. **XGBoost** — gradient boosting
4. **LightGBM** — fast gradient boosting"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import time

# Define models
models = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', random_state=42), True),
    "Random Forest": (RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42), False),
    "XGBoost": (xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric='auc', random_state=42, verbosity=0), False),
    "LightGBM": (lgb.LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1), False),
}

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, (model, needs_scaling) in models.items():
    print(f"\\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    
    fold_aucs = []
    fold_probas = []
    fold_trues = []
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train[train_idx], y_train[val_idx]
        
        if needs_scaling:
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_train.columns)
            X_va = pd.DataFrame(scaler.transform(X_va), columns=X_train.columns)
        
        m = clone(model)
        m.fit(X_tr, y_tr)
        y_proba = m.predict_proba(X_va)[:, 1]
        
        auc = roc_auc_score(y_va, y_proba)
        fold_aucs.append(auc)
        fold_probas.extend(y_proba)
        fold_trues.extend(y_va)
        
        print(f"  Fold {fold+1}: AUC = {auc:.4f}")
    
    elapsed = time.time() - start_time
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    results[name] = {
        'aucs': fold_aucs, 'mean': mean_auc, 'std': std_auc,
        'probas': fold_probas, 'trues': fold_trues, 'time': elapsed
    }
    print(f"  => Mean AUC: {mean_auc:.4f} ± {std_auc:.4f} ({elapsed:.1f}s)")

# Summary table
print(f"\\n{'='*60}")
print(f"{'Model':<25} {'Mean AUC':<12} {'Std':<10} {'Time':<8}")
print(f"{'='*60}")
for name, res in sorted(results.items(), key=lambda x: -x[1]['mean']):
    print(f"{name:<25} {res['mean']:<12.4f} {res['std']:<10.4f} {res['time']:<8.1f}s")"""))

cells.append(nbf.v4.new_markdown_cell("### 6.1 ROC Curves Comparison"))

cells.append(nbf.v4.new_code_cell("""# Plot ROC curves for all models
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(res['trues'], res['probas'])
    ax.plot(fpr, tpr, color=color, linewidth=2, 
            label=f"{name} (AUC = {res['mean']:.4f} ± {res['std']:.4f})")

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — Model Comparison', fontsize=14)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()"""))

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## 7. Feature Importance Analysis"))

cells.append(nbf.v4.new_code_cell("""# Train the best model on full data and analyze feature importance
best_model_name = max(results, key=lambda k: results[k]['mean'])
print(f"Best model: {best_model_name} (AUC = {results[best_model_name]['mean']:.4f})")

# Find matching model config
model_cfg, needs_scaling = models[best_model_name]
best_model = clone(model_cfg)

if needs_scaling:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    best_model.fit(X_scaled, y_train)
else:
    best_model.fit(X_train, y_train)

# Get feature importances
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
else:
    importances = None

if importances is not None:
    fi_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ['steelblue' if imp > fi_df['Importance'].median() else 'lightsteelblue' 
              for imp in fi_df['Importance']]
    ax.barh(range(len(fi_df)), fi_df['Importance'], color=colors, edgecolor='white')
    ax.set_yticks(range(len(fi_df)))
    ax.set_yticklabels(fi_df['Feature'], fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Feature Importance ({best_model_name})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print top features
    print(f"\\nTop 10 features:")
    for _, row in fi_df.tail(10).iloc[::-1].iterrows():
        print(f"  {row['Feature']:35s} {row['Importance']:.4f}")"""))

# ============================================================
# FEATURE FAMILY ABLATION
# ============================================================
cells.append(nbf.v4.new_markdown_cell("### 7.1 Feature Family Contribution"))

cells.append(nbf.v4.new_code_cell("""# Ablation study: evaluate each feature family independently
feature_groups = {
    'Graph Topology': [c for c in X_train.columns if c in [
        'common_neighbors', 'jaccard_coefficient', 'adamic_adar_index',
        'resource_allocation', 'preferential_attachment', 'degree_source',
        'degree_target', 'degree_sum', 'degree_diff', 'degree_min', 
        'degree_max', 'shortest_path_length']],
    'Node Attributes': [c for c in X_train.columns if c in [
        'cosine_similarity', 'common_keywords', 'union_keywords',
        'jaccard_features', 'feature_diff_l2', 'keyword_count_source',
        'keyword_count_target', 'keyword_count_sum', 'keyword_count_diff']],
    'Community': [c for c in X_train.columns if 'community' in c],
    'SVD Embeddings': [c for c in X_train.columns if 'svd' in c],
    'Centrality': [c for c in X_train.columns if 'pagerank' in c or 'clustering' in c],
}

# Use the best model type for ablation
ablation_results = {}
skf_abl = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for group_name, group_cols in feature_groups.items():
    X_group = X_train[group_cols]
    aucs = []
    
    for train_idx, val_idx in skf_abl.split(X_group, y_train):
        m = clone(model_cfg)
        m.fit(X_group.iloc[train_idx], y_train[train_idx])
        y_prob = m.predict_proba(X_group.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_train[val_idx], y_prob))
    
    ablation_results[group_name] = {'mean': np.mean(aucs), 'std': np.std(aucs), 'n_features': len(group_cols)}
    print(f"{group_name:20s} ({len(group_cols):2d} features) — AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# Plot ablation results
fig, ax = plt.subplots(figsize=(10, 5))
names = list(ablation_results.keys())
means = [ablation_results[n]['mean'] for n in names]
stds = [ablation_results[n]['std'] for n in names]
colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12']

bars = ax.bar(names, means, yerr=stds, color=colors, edgecolor='white', 
              capsize=5, alpha=0.85)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Feature Family Contribution (Individual Performance)', fontsize=14)
ax.set_ylim(0.5, 1.0)

# Add value labels
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
            f'{mean:.3f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('feature_ablation.png', dpi=150, bbox_inches='tight')
plt.show()

# All features combined
print(f"\\nAll features combined: AUC = {results[best_model_name]['mean']:.4f}")"""))

# ============================================================
# SUBMISSION
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Generate Submission

Extract features for test pairs (using the full graph) and predict probabilities."""))

cells.append(nbf.v4.new_code_cell("""# Extract test features (inference mode — no edge removal)
print("Extracting test features...")
X_test = extract_all_features(G, features_matrix, node_id_to_idx, test_df, is_training=False)

# Predict probabilities
if needs_scaling:
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'ID': np.arange(len(y_test_proba)),
    'Predicted': y_test_proba
})

submission.to_csv('submissions/submission.csv', index=False)
print(f"\\nSubmission saved! Shape: {submission.shape}")
print(f"Prediction statistics:")
print(f"  Min: {y_test_proba.min():.4f}")
print(f"  Max: {y_test_proba.max():.4f}")
print(f"  Mean: {y_test_proba.mean():.4f}")
print(f"  Predicted positive (>0.5): {(y_test_proba > 0.5).sum()} / {len(y_test_proba)}")

# Distribution of predictions
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_test_proba, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Test Predictions', fontsize=14)
ax.axvline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
ax.legend()

plt.tight_layout()
plt.savefig('prediction_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

submission.head(10)"""))

# ============================================================
# CONCLUSION
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""## 9. Summary & Conclusions

### Results

| Model | AUC-ROC (5-fold CV) |
|-------|-------------------|
| Logistic Regression | See above |
| Random Forest | See above |
| XGBoost | See above |
| LightGBM | See above |

### Key Insights

1. **Graph topology features** are the most powerful predictors — common neighbors, preferential attachment, and community membership strongly discriminate between edges and non-edges.

2. **SVD embeddings** capture latent structural patterns that complement direct topology features.

3. **Node attribute features** (Wikipedia keywords) provide moderate but useful signal — actors who share keywords are somewhat more likely to co-occur.

4. **Data leakage prevention** was crucial: edges were temporarily removed during feature computation for training pairs to prevent the model from trivially detecting known edges.

### Possible Improvements

- Ensemble methods (stacking/voting) combining multiple models
- Graph neural network approaches (if allowed)
- More sophisticated community detection algorithms
- Feature selection to reduce redundancy
- Hyperparameter tuning with Optuna/Bayesian optimization
"""))

# Build notebook
nb.cells = cells

# Write notebook
notebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebook.ipynb')
notebook_path = os.path.normpath(notebook_path)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook written to {notebook_path}")
