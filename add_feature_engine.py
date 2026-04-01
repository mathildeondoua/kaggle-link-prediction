import nbformat as nbf
from nbformat.v4 import new_markdown_cell, new_code_cell

# Read the notebook
with open('main.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

nb.cells.extend([
    new_markdown_cell("### 3. Feature Engineering"),
    new_markdown_cell("Based on the EDA, the graph is highly sparse but connected (power-law distribution/scale free). \nTherefore graph topology features (Adamic Adar, Jaccard, Preferential Attachment) will be very discriminative.\nAlso, since node features are 99.4% zeros, we'll use fast sparse computations for text features (Cosine, Dot product, etc.)."),
    new_code_cell("""import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity, paired_distances

# Helper function to generate features for a list of edges
def build_features(edges, G, feature_matrix):
    n_edges = len(edges)
    features = np.zeros((n_edges, 10)) # Adjust size as we add features
    
    # Convert feature matrix to sparse for efficiency
    sparse_features = sp.csr_matrix(feature_matrix)
    
    # Pre-extract arrays for pairwise computation
    sources = [e[0] for e in edges]
    targets = [e[1] for e in edges]
    f_sources = sparse_features[sources]
    f_targets = sparse_features[targets]
    
    # 1. Cosine Similarity
    cos_sim = np.array(np.sum(f_sources.multiply(f_targets), axis=1)).flatten() / (
        np.linalg.norm(f_sources.toarray(), axis=1) * np.linalg.norm(f_targets.toarray(), axis=1) + 1e-10)
        
    # 2. Dot Product (Overlap of active keywords)
    dot_prod = np.array(np.sum(f_sources.multiply(f_targets), axis=1)).flatten()
    
    # 3. L2 Distance
    l2_dist = np.linalg.norm(f_sources.toarray() - f_targets.toarray(), axis=1)
    
    # Loop for graph-based features
    jaccard = []
    adamic_adar = []
    pref_attach = []
    common_neighbors = []
    shortest_path = []
    
    for u, v in tqdm(edges, desc="Computing Graph Features"):
        # Jaccard
        try:
            preds = nx.jaccard_coefficient(G, [(u, v)])
            jaccard.append(next(preds)[2])
        except ZeroDivisionError:
            jaccard.append(0)
            
        # Adamic Adar
        try:
            preds = nx.adamic_adar_index(G, [(u, v)])
            adamic_adar.append(next(preds)[2])
        except (ZeroDivisionError, nx.NetworkXError):
            adamic_adar.append(0)
            
        # Preferential Attachment
        preds = nx.preferential_attachment(G, [(u, v)])
        pref_attach.append(next(preds)[2])
        
        # Common neighbors
        cn = list(nx.common_neighbors(G, u, v))
        common_neighbors.append(len(cn))
        
        # Shortest path (max 5)
        try:
            length = nx.shortest_path_length(G, source=u, target=v)
            shortest_path.append(length if length <= 5 else 6)
        except nx.NetworkXNoPath:
            shortest_path.append(-1)
            
    features[:, 0] = cos_sim
    features[:, 1] = dot_prod
    features[:, 2] = l2_dist
    features[:, 3] = jaccard
    features[:, 4] = adamic_adar
    features[:, 5] = pref_attach
    features[:, 6] = common_neighbors
    features[:, 7] = shortest_path
    
    # Keyword Frequency weighting (TF-IDF like feature)
    # Rare keywords matching
    keyword_freq = np.array(sparse_features.sum(axis=0)).flatten()
    idf = np.log((sparse_features.shape[0] + 1) / (keyword_freq + 1))
    
    intersect = f_sources.multiply(f_targets)
    weighted_overlap = intersect.multiply(idf).sum(axis=1)
    features[:, 8] = np.array(weighted_overlap).flatten()
    
    # Degree diff
    deg_diff = []
    for u, v in edges:
        deg_diff.append(abs(G.degree(u) - G.degree(v)))
    features[:, 9] = deg_diff
    
    return features

# Generate Features
X_train = build_features(train_edges, G_train, feature_matrix)
y_train = train_df['label'].values

X_test = build_features(test_edges, G_train, feature_matrix)

feature_names = [
    "Cosine_Sim", "Dot_Product", "L2_Dist",
    "Jaccard", "Adamic_Adar", "Pref_Attach", "Common_Neighbors",
    "Shortest_Path", "Weighted_Rare_Overlap", "Degree_Diff"
]

print(f"Successfully generated train features: {X_train.shape}")
print(f"Successfully generated test features: {X_test.shape}")
"""),
    new_markdown_cell("### 4. Baseline Model & Feature Importance"),
    new_code_cell("""from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Handle any Infs or NaNs
X_train = np.nan_to_num(X_train, posinf=0, neginf=0)
X_test = np.nan_to_num(X_test, posinf=0, neginf=0)

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"Baseline Random Forest CV ROC-AUC: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

# Train a model on everything to plot feature importance
rf.fit(X_train, y_train)

# Plot Feature Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
""")
])

with open('main.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Added feature engineering cells to main.ipynb")
