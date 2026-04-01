import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

nb.cells.extend([
    new_markdown_cell("# Kaggle Link Prediction\n## Phase 0 & 1: Data Loading and EDA"),
    new_code_cell("""import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from tqdm.auto import tqdm
import scipy.sparse as sp

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
%matplotlib inline
"""),
    new_markdown_cell("### 1. Data Loading"),
    new_code_cell("""# Load training edges
train_edges = []
train_labels = []
with open('data/train.txt', 'r') as f:
    for line in f:
        u, v, label = line.strip().split()
        train_edges.append((int(u), int(v)))
        train_labels.append(int(label))

train_df = pd.DataFrame(train_edges, columns=['source', 'target'])
train_df['label'] = train_labels

# Load testing edges
test_edges = []
with open('data/test.txt', 'r') as f:
    for line in f:
        u, v = line.strip().split()
        test_edges.append((int(u), int(v)))

test_df = pd.DataFrame(test_edges, columns=['source', 'target'])

# Load node features
node_features = pd.read_csv('data/node_information.csv', header=None)
node_ids = node_features[0].values
feature_matrix = node_features.drop(0, axis=1).values

print(f"Train pairs: {len(train_df)}")
print(f"Test pairs: {len(test_df)}")
print(f"Class balance in Train (Positive edges): {train_df['label'].mean():.2%}")
print(f"Node features shape: {feature_matrix.shape}")
print(f"Sparsity of feature matrix: {(feature_matrix == 0).sum() / feature_matrix.size:.2%}")
"""),
    new_markdown_cell("### 2. Exploratory Graph Analysis"),
    new_code_cell("""# Build the training graph using ONLY positive labels
positive_edges = train_df[train_df['label'] == 1][['source', 'target']].values
G_train = nx.Graph()
G_train.add_edges_from(positive_edges)

print(f"Graph Nodes: {G_train.number_of_nodes()}")
print(f"Graph Edges: {G_train.number_of_edges()}")
print(f"Connected Components: {nx.number_connected_components(G_train)}")
largest_cc = max(nx.connected_components(G_train), key=len)
print(f"Largest Component Size: {len(largest_cc)} nodes")

# Degree Distribution
degrees = [deg for node, deg in G_train.degree()]
plt.figure(figsize=(10, 5))
plt.hist(degrees, bins=50, color='skyblue', edgecolor='black')
plt.title('Degree Distribution (Linear)')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.show()

# Log-Log Degree Distribution (Check for Scale-Free)
degree_counts = pd.Series(degrees).value_counts().sort_index()
plt.figure(figsize=(10, 5))
plt.scatter(degree_counts.index, degree_counts.values, alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.title('Degree Distribution (Log-Log Scale)')
plt.xlabel('Log Degree')
plt.ylabel('Log Count')
plt.show()
""")
])

with open('main.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook main.ipynb created successfully.")
