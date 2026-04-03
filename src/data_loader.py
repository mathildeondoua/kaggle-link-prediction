"""
data_loader.py — Data loading and graph construction for link prediction.

Provides functions to load node features, train/test pairs, 
and build a NetworkX graph from the known positive edges.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx


# Default data directory (relative to project root)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_node_features(data_dir=DATA_DIR):
    """
    Load node features from node_information.csv.
    
    Returns:
        node_ids: np.array of node IDs
        features_matrix: np.array of shape (n_nodes, 932), binary features
        node_id_to_idx: dict mapping node_id -> row index in features_matrix
    """
    filepath = os.path.join(data_dir, "node_information.csv")
    raw = pd.read_csv(filepath, header=None)
    
    node_ids = raw.iloc[:, 0].values.astype(int)
    features_matrix = raw.iloc[:, 1:].values.astype(np.float32)
    
    # Build lookup: node_id -> row index for fast access
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    
    print(f"[data_loader] Loaded {len(node_ids)} nodes with {features_matrix.shape[1]} features each.")
    return node_ids, features_matrix, node_id_to_idx


def load_train(data_dir=DATA_DIR):
    """
    Load the training set of labeled node pairs.
    
    Returns:
        DataFrame with columns: source, target, label
    """
    filepath = os.path.join(data_dir, "train.txt")
    train_df = pd.read_csv(filepath, sep=" ", header=None, names=["source", "target", "label"])
    print(f"[data_loader] Loaded {len(train_df)} training pairs "
          f"(pos={train_df['label'].sum()}, neg={len(train_df) - train_df['label'].sum()}).")
    return train_df


def load_test(data_dir=DATA_DIR):
    """
    Load the test set of unlabeled node pairs.
    
    Returns:
        DataFrame with columns: source, target
    """
    filepath = os.path.join(data_dir, "test.txt")
    test_df = pd.read_csv(filepath, sep=" ", header=None, names=["source", "target"])
    print(f"[data_loader] Loaded {len(test_df)} test pairs.")
    return test_df


def build_graph(train_df, node_ids):
    """
    Build a NetworkX graph from the positive edges in the training set.
    
    All nodes from node_information are added (even if they have no edge),
    and only edges with label=1 in train are added.
    
    Args:
        train_df: DataFrame with columns source, target, label
        node_ids: array of all node IDs
    
    Returns:
        G: NetworkX undirected graph
    """
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    
    positive_edges = train_df[train_df["label"] == 1]
    G.add_edges_from(zip(positive_edges["source"], positive_edges["target"]))
    
    print(f"[data_loader] Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


if __name__ == "__main__":
    # Quick sanity check
    node_ids, features_matrix, node_id_to_idx = load_node_features()
    train_df = load_train()
    test_df = load_test()
    G = build_graph(train_df, node_ids)
    
    print(f"\nSanity checks:")
    print(f"  Features shape: {features_matrix.shape}")
    print(f"  Train label balance: {train_df['label'].value_counts().to_dict()}")
    print(f"  Graph density: {nx.density(G):.6f}")
    print(f"  Connected components: {nx.number_connected_components(G)}")
