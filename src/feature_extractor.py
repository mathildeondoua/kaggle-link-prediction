"""
feature_extractor.py — Feature engineering for link prediction.

Extracts graph-based, node-attribute-based, and community-based features
for pairs of nodes. This is the core of the pipeline: good features = good AUC.

IMPORTANT: For training pairs, the edge (u,v) must be removed from the graph
before computing graph features to avoid data leakage. The extract_all_features 
function handles this via the 'is_training' flag.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def _safe_shortest_path(G, source, target, cutoff=6):
    """
    Compute shortest path length between source and target.
    Returns cutoff+1 if no path exists within cutoff hops.
    """
    try:
        return nx.shortest_path_length(G, source, target)
    except nx.NetworkXNoPath:
        return cutoff + 1
    except nx.NodeNotFound:
        return cutoff + 1


def compute_graph_features(G, pairs_df, remove_edges=False):
    """
    Compute topological / graph-based features for each node pair.
    
    When remove_edges=True (training mode), temporarily removes each edge (u,v)
    from the graph before computing features for that pair to avoid leakage.
    
    Features computed:
        - common_neighbors: count of shared neighbors
        - jaccard_coefficient: |common| / |union| of neighbors
        - adamic_adar_index: sum of 1/log(deg(w)) for common neighbors w
        - resource_allocation: sum of 1/deg(w) for common neighbors w
        - preferential_attachment: deg(u) * deg(v)
        - degree_source, degree_target: individual degrees
        - degree_sum, degree_diff, degree_min, degree_max: degree combinations
        - shortest_path_length: shortest path distance (capped)
    
    Args:
        G: NetworkX undirected graph
        pairs_df: DataFrame with 'source' and 'target' columns
        remove_edges: if True, remove edge (s,t) before computing features
    
    Returns:
        DataFrame with graph features (one row per pair)
    """
    sources = pairs_df["source"].values
    targets = pairs_df["target"].values
    n_pairs = len(pairs_df)
    
    # Initialize feature arrays
    common_neighbors = np.zeros(n_pairs, dtype=np.float32)
    jaccard_coeff = np.zeros(n_pairs, dtype=np.float32)
    adamic_adar = np.zeros(n_pairs, dtype=np.float32)
    resource_alloc = np.zeros(n_pairs, dtype=np.float32)
    pref_attachment = np.zeros(n_pairs, dtype=np.float32)
    deg_source = np.zeros(n_pairs, dtype=np.float32)
    deg_target = np.zeros(n_pairs, dtype=np.float32)
    spl = np.zeros(n_pairs, dtype=np.float32)
    
    for i in range(n_pairs):
        s, t = sources[i], targets[i]
        
        # Temporarily remove the edge (s,t) if it exists and we are in training mode
        edge_existed = False
        if remove_edges and G.has_edge(s, t):
            G.remove_edge(s, t)
            edge_existed = True
        
        neighbors_s = set(G.neighbors(s)) if G.has_node(s) else set()
        neighbors_t = set(G.neighbors(t)) if G.has_node(t) else set()
        
        common = neighbors_s & neighbors_t
        union = neighbors_s | neighbors_t
        
        # Common neighbors count
        cn = len(common)
        common_neighbors[i] = cn
        
        # Jaccard coefficient
        if len(union) > 0:
            jaccard_coeff[i] = cn / len(union)
        
        # Adamic-Adar index & Resource Allocation
        aa = 0.0
        ra = 0.0
        for w in common:
            deg_w = G.degree(w)
            if deg_w > 1:
                aa += 1.0 / np.log(deg_w)
            ra += 1.0 / deg_w if deg_w > 0 else 0.0
        adamic_adar[i] = aa
        resource_alloc[i] = ra
        
        # Degrees and preferential attachment
        ds = G.degree(s) if G.has_node(s) else 0
        dt = G.degree(t) if G.has_node(t) else 0
        pref_attachment[i] = ds * dt
        deg_source[i] = ds
        deg_target[i] = dt
        
        # Shortest path length
        spl[i] = _safe_shortest_path(G, s, t, cutoff=6)
        
        # Restore the edge if we removed it
        if edge_existed:
            G.add_edge(s, t)
    
    features = pd.DataFrame({
        "common_neighbors": common_neighbors,
        "jaccard_coefficient": jaccard_coeff,
        "adamic_adar_index": adamic_adar,
        "resource_allocation": resource_alloc,
        "preferential_attachment": pref_attachment,
        "degree_source": deg_source,
        "degree_target": deg_target,
        "degree_sum": deg_source + deg_target,
        "degree_diff": np.abs(deg_source - deg_target),
        "degree_min": np.minimum(deg_source, deg_target),
        "degree_max": np.maximum(deg_source, deg_target),
        "shortest_path_length": spl,
    })
    
    print(f"[feature_extractor] Computed {features.shape[1]} graph features for {n_pairs} pairs.")
    return features


def compute_node_feature_similarities(features_matrix, node_id_to_idx, pairs_df):
    """
    Compute similarity features based on binary node attributes (Wikipedia keywords).
    
    Features computed:
        - cosine_similarity: cosine between feature vectors
        - common_keywords: count of shared keywords (both = 1)
        - union_keywords: count of union keywords (either = 1)
        - jaccard_features: common / union of keywords
        - feature_diff_l2: L2 norm of difference
        - keyword_count_source, keyword_count_target: per-node keyword counts
        - keyword_count_sum, keyword_count_diff: combinations
    """
    sources = pairs_df["source"].values
    targets = pairs_df["target"].values
    n_pairs = len(pairs_df)
    
    # Pre-fetch feature vectors
    source_indices = np.array([node_id_to_idx[s] for s in sources])
    target_indices = np.array([node_id_to_idx[t] for t in targets])
    
    feat_s = features_matrix[source_indices]  # (n_pairs, 932)
    feat_t = features_matrix[target_indices]  # (n_pairs, 932)
    
    # Keyword counts per node
    kw_count_s = feat_s.sum(axis=1)
    kw_count_t = feat_t.sum(axis=1)
    
    # Common keywords (both = 1)
    common_kw = (feat_s * feat_t).sum(axis=1)
    
    # Union keywords (either = 1)
    union_kw = ((feat_s + feat_t) > 0).sum(axis=1).astype(np.float32)
    
    # Jaccard on features
    jaccard_feat = np.where(union_kw > 0, common_kw / union_kw, 0.0)
    
    # Cosine similarity
    norm_s = np.linalg.norm(feat_s, axis=1)
    norm_t = np.linalg.norm(feat_t, axis=1)
    denom = norm_s * norm_t
    cosine_sim = np.where(denom > 0, (feat_s * feat_t).sum(axis=1) / denom, 0.0)
    
    # L2 distance
    diff_l2 = np.linalg.norm(feat_s - feat_t, axis=1)
    
    features = pd.DataFrame({
        "cosine_similarity": cosine_sim.astype(np.float32),
        "common_keywords": common_kw.astype(np.float32),
        "union_keywords": union_kw,
        "jaccard_features": jaccard_feat.astype(np.float32),
        "feature_diff_l2": diff_l2.astype(np.float32),
        "keyword_count_source": kw_count_s.astype(np.float32),
        "keyword_count_target": kw_count_t.astype(np.float32),
        "keyword_count_sum": (kw_count_s + kw_count_t).astype(np.float32),
        "keyword_count_diff": np.abs(kw_count_s - kw_count_t).astype(np.float32),
    })
    
    print(f"[feature_extractor] Computed {features.shape[1]} node-attribute features for {n_pairs} pairs.")
    return features


def compute_community_features(G, pairs_df):
    """
    Compute community-based features using Louvain community detection.
    
    Features computed:
        - same_community: 1 if both nodes are in same community
        - community_size_source, community_size_target: community sizes
    """
    print("[feature_extractor] Running Louvain community detection...")
    communities = nx.community.louvain_communities(G, seed=42)
    
    node_to_community = {}
    community_sizes = {}
    for cid, members in enumerate(communities):
        community_sizes[cid] = len(members)
        for node in members:
            node_to_community[node] = cid
    
    print(f"[feature_extractor] Found {len(communities)} communities.")
    
    sources = pairs_df["source"].values
    targets = pairs_df["target"].values
    
    comm_s = np.array([node_to_community.get(s, -1) for s in sources])
    comm_t = np.array([node_to_community.get(t, -1) for t in targets])
    
    same_community = (comm_s == comm_t).astype(np.float32)
    comm_size_s = np.array([community_sizes.get(c, 0) for c in comm_s], dtype=np.float32)
    comm_size_t = np.array([community_sizes.get(c, 0) for c in comm_t], dtype=np.float32)
    
    features = pd.DataFrame({
        "same_community": same_community,
        "community_size_source": comm_size_s,
        "community_size_target": comm_size_t,
    })
    
    print(f"[feature_extractor] Computed {features.shape[1]} community features.")
    return features


def compute_svd_features(G, node_id_to_idx, pairs_df, n_components=64):
    """
    Compute SVD-based graph embedding features.
    
    Uses truncated SVD on the adjacency matrix to capture latent structure.
    Computes dot product, distance, and cosine similarity between embeddings.
    """
    print(f"[feature_extractor] Computing SVD embeddings (k={n_components})...")
    
    n_nodes = len(node_id_to_idx)
    
    # Build sparse adjacency matrix
    rows, cols = [], []
    for u, v in G.edges():
        if u in node_id_to_idx and v in node_id_to_idx:
            idx_u = node_id_to_idx[u]
            idx_v = node_id_to_idx[v]
            rows.extend([idx_u, idx_v])
            cols.extend([idx_v, idx_u])
    
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))
    
    # Truncated SVD
    U, S, Vt = svds(adj.astype(np.float64), k=n_components)
    
    # Node embeddings: U * sqrt(S)
    embeddings = U * np.sqrt(S)
    
    sources = pairs_df["source"].values
    targets = pairs_df["target"].values
    
    source_indices = np.array([node_id_to_idx[s] for s in sources])
    target_indices = np.array([node_id_to_idx[t] for t in targets])
    
    emb_s = embeddings[source_indices]
    emb_t = embeddings[target_indices]
    
    # Dot product captures structural similarity
    svd_dot = (emb_s * emb_t).sum(axis=1)
    
    # Euclidean distance
    svd_dist = np.linalg.norm(emb_s - emb_t, axis=1)
    
    # Cosine similarity in SVD space
    norm_s = np.linalg.norm(emb_s, axis=1)
    norm_t = np.linalg.norm(emb_t, axis=1)
    denom = norm_s * norm_t
    svd_cosine = np.where(denom > 0, svd_dot / denom, 0.0)
    
    features = pd.DataFrame({
        "svd_dot_product": svd_dot.astype(np.float32),
        "svd_distance": svd_dist.astype(np.float32),
        "svd_cosine_similarity": svd_cosine.astype(np.float32),
    })
    
    print(f"[feature_extractor] Computed {features.shape[1]} SVD features.")
    return features


def compute_centrality_features(G, pairs_df):
    """
    Compute centrality-based features for each node.
    
    Features: PageRank and clustering coefficients with combinations.
    """
    print("[feature_extractor] Computing PageRank...")
    pagerank = nx.pagerank(G)
    
    print("[feature_extractor] Computing clustering coefficients...")
    clustering = nx.clustering(G)
    
    sources = pairs_df["source"].values
    targets = pairs_df["target"].values
    
    pr_s = np.array([pagerank.get(s, 0.0) for s in sources], dtype=np.float32)
    pr_t = np.array([pagerank.get(t, 0.0) for t in targets], dtype=np.float32)
    cc_s = np.array([clustering.get(s, 0.0) for s in sources], dtype=np.float32)
    cc_t = np.array([clustering.get(t, 0.0) for t in targets], dtype=np.float32)
    
    features = pd.DataFrame({
        "pagerank_source": pr_s,
        "pagerank_target": pr_t,
        "pagerank_sum": pr_s + pr_t,
        "pagerank_diff": np.abs(pr_s - pr_t),
        "pagerank_product": pr_s * pr_t,
        "clustering_coeff_source": cc_s,
        "clustering_coeff_target": cc_t,
        "clustering_coeff_sum": cc_s + cc_t,
        "clustering_coeff_diff": np.abs(cc_s - cc_t),
    })
    
    print(f"[feature_extractor] Computed {features.shape[1]} centrality features.")
    return features


def extract_all_features(G, features_matrix, node_id_to_idx, pairs_df, is_training=False):
    """
    Extract all feature families and concatenate them into a single DataFrame.
    
    IMPORTANT: When is_training=True, graph features are computed by temporarily
    removing the edge (u,v) before computing features for each pair. This prevents
    data leakage since positive training pairs are by definition edges in the graph.
    
    For test pairs (is_training=False), the full graph is used.
    
    Args:
        G: NetworkX undirected graph
        features_matrix: np.array (n_nodes, 932) binary node features
        node_id_to_idx: dict mapping node_id -> row index
        pairs_df: DataFrame with 'source' and 'target' columns
        is_training: if True, remove edges to prevent leakage
    
    Returns:
        DataFrame with all features concatenated (one row per pair)
    """
    mode = "TRAINING (edge removal)" if is_training else "INFERENCE (full graph)"
    print(f"\n{'='*60}")
    print(f"[feature_extractor] Extracting features for {len(pairs_df)} pairs...")
    print(f"[feature_extractor] Mode: {mode}")
    print(f"{'='*60}\n")
    
    # 1. Graph topological features (with edge removal for training)
    graph_feats = compute_graph_features(G, pairs_df, remove_edges=is_training)
    
    # 2. Node attribute similarity features (no leakage risk)
    attr_feats = compute_node_feature_similarities(features_matrix, node_id_to_idx, pairs_df)
    
    # 3. Community features (computed on full graph — acceptable since
    #    community structure is a global property, not pair-specific)
    community_feats = compute_community_features(G, pairs_df)
    
    # 4. SVD embedding features (computed on full graph — same rationale)
    svd_feats = compute_svd_features(G, node_id_to_idx, pairs_df, n_components=64)
    
    # 5. Centrality features (computed on full graph)
    centrality_feats = compute_centrality_features(G, pairs_df)
    
    # Concatenate all feature families
    all_features = pd.concat(
        [graph_feats, attr_feats, community_feats, svd_feats, centrality_feats],
        axis=1
    )
    
    print(f"\n[feature_extractor] Total: {all_features.shape[1]} features extracted.")
    
    return all_features


if __name__ == "__main__":
    # Quick test: extract features for a small sample of train pairs
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_node_features, load_train, build_graph
    
    node_ids, features_matrix, node_id_to_idx = load_node_features()
    train_df = load_train()
    G = build_graph(train_df, node_ids)
    
    # Test on first 100 pairs WITH edge removal (training mode)
    sample = train_df.head(100)
    feats = extract_all_features(G, features_matrix, node_id_to_idx, sample, is_training=True)
    
    print(f"\nSample features shape: {feats.shape}")
    print(f"\nFeature statistics:\n{feats.describe().T[['mean', 'std', 'min', 'max']]}")
    
    # Verify no leakage: shortest_path should NOT always be 1 for positive pairs
    pos_sample = sample[sample["label"] == 1]
    pos_feats = feats.iloc[pos_sample.index]
    print(f"\nLeakage check — Positive pairs shortest_path distribution:")
    print(pos_feats["shortest_path_length"].value_counts().sort_index())
    
    # Check for NaN/Inf
    print(f"\nNaN count: {feats.isna().sum().sum()}")
    print(f"Inf count: {np.isinf(feats.values).sum()}")
