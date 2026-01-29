
import os
import hashlib
import torch
import traceback

import torch.nn as nn
import networkx as nx
import numpy as np


from collections import defaultdict
from typing import Tuple, Dict, List, Iterable


from torch_geometric.data import Data
from torch_geometric.utils import degree


from typing import Tuple, Dict, List, Iterable


import matplotlib.pyplot as plt


# --------------------------
#  Dataset Analysis
# --------------------------
def analyze_dataset_statistics(dataset: Iterable[Data], num_classes: int) -> Tuple[Dict, Dict]:
    """Analyze dataset statistics (per-class node/edge counts)."""
    class_node_stats = defaultdict(list)
    class_edge_stats = defaultdict(list)

    for graph in dataset:
        label = int(graph.y.item())
        num_nodes = graph.x.size(0)
        # count undirected edges as edge_index stores both directions in many TU datasets
        num_edges = graph.edge_index.size(1) // 2 if graph.edge_index.size(1) > 0 else 0

        class_node_stats[label].append(num_nodes)
        class_edge_stats[label].append(num_edges)

    node_stats_summary = {}
    edge_stats_summary = {}

    for class_id in range(num_classes):
        if class_id in class_node_stats:
            nodes = np.array(class_node_stats[class_id])
            edges = np.array(class_edge_stats[class_id])

            node_stats_summary[class_id] = {
                'min': int(nodes.min()),
                'max': int(nodes.max()),
                'mean': float(nodes.mean()),
                'std': float(nodes.std())
            }

            edge_stats_summary[class_id] = {
                'min': int(edges.min()),
                'max': int(edges.max()),
                'mean': float(edges.mean()),
                'std': float(edges.std())
            }

    return node_stats_summary, edge_stats_summary


# --------------------------
#  Graph Statistics and MMD
# --------------------------
def degree_distribution(graph: Data, num_bins: int = 10):
    """Compute degree histogram for a graph."""
    if graph.num_nodes == 0:
        return np.zeros(num_bins)

    row = graph.edge_index[0]
    if row.numel() == 0:
        return np.zeros(num_bins)

    deg = degree(row, num_nodes=graph.num_nodes).cpu().numpy()
    if len(deg) == 0:
        return np.zeros(num_bins)

    hist, _ = np.histogram(deg, bins=num_bins, range=(0, max(int(deg.max()), num_bins)))
    return hist / hist.sum() if hist.sum() > 0 else hist


def clustering_coefficient_distribution(graph: Data, num_bins: int = 10):
    """Compute clustering coefficient distribution using NetworkX."""
    if graph.num_nodes < 3:
        return np.zeros(num_bins)

    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    edges = graph.edge_index.cpu().numpy().T
    if edges.size > 0:
        G.add_edges_from(edges)

    clustering = list(nx.clustering(G).values())
    if len(clustering) == 0:
        return np.zeros(num_bins)

    hist, _ = np.histogram(clustering, bins=num_bins, range=(0, 1))
    return hist / hist.sum() if hist.sum() > 0 else hist


def spectral_features(graph: Data, num_features: int = 10):
    """Compute spectral features (first num_features Laplacian eigenvalues)."""
    if graph.num_nodes < 2:
        return np.zeros(num_features)

    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    edges = graph.edge_index.cpu().numpy().T
    if edges.size > 0:
        G.add_edges_from(edges)

    try:
        laplacian_eigenvalues = np.array(nx.laplacian_spectrum(G))
        features = np.zeros(num_features)
        k = min(len(laplacian_eigenvalues), num_features)
        features[:k] = laplacian_eigenvalues[:k]
        return features
    except Exception as e:
        print("spectral_features() failed with exception:", e)
        #traceback.print_exc()
        return np.zeros(num_features)



def compute_mmd(X, Y, kernel: str = 'rbf', gamma: float = 1.0):
    """Compute Maximum Mean Discrepancy between two sets of feature vectors."""
    if len(X) == 0 or len(Y) == 0:
        return 0.0

    X = np.array(X) if isinstance(X, list) else X
    Y = np.array(Y) if isinstance(Y, list) else Y

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    m, n = X.size(0), Y.size(0)
    if m == 0 or n == 0:
        return 0.0

    # Normalize features per-dimension
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-8
    X = (X - X_mean) / X_std
    Y = (Y - Y_mean) / Y_std

    if kernel == 'rbf':
        XX = torch.mm(X, X.t())
        YY = torch.mm(Y, Y.t())
        XY = torch.mm(X, Y.t())

        X_sqnorms = torch.diag(XX).unsqueeze(1)
        Y_sqnorms = torch.diag(YY).unsqueeze(1)

        K_XX = torch.exp(-gamma * (X_sqnorms + X_sqnorms.t() - 2 * XX))
        K_YY = torch.exp(-gamma * (Y_sqnorms + Y_sqnorms.t() - 2 * YY))
        K_XY = torch.exp(-gamma * (X_sqnorms + Y_sqnorms.t() - 2 * XY))
    else:
        K_XX = torch.mm(X, X.t())
        K_YY = torch.mm(Y, Y.t())
        K_XY = torch.mm(X, Y.t())

    mmd = K_XX.sum() / (m * m) + K_YY.sum() / (n * n) - 2 * K_XY.sum() / (m * n)
    return float(mmd.item())


def compute_graph_statistics(graphs: List[Data], num_bins: int = 10):
    """Compute comprehensive statistics for a list of graphs."""
    if len(graphs) == 0:
        return {}

    n_graphs = len(graphs)

    num_nodes = np.zeros(n_graphs)
    num_edges = np.zeros(n_graphs)
    degrees = np.zeros((n_graphs, num_bins))
    clustering = np.zeros((n_graphs, num_bins))
    spectral = np.zeros((n_graphs, num_bins))

    for i, g in enumerate(graphs):
        num_nodes[i] = g.x.size(0)
        num_edges[i] = g.edge_index.size(1) // 2 if g.edge_index.size(1) > 0 else 0
        degrees[i] = degree_distribution(g, num_bins)
        clustering[i] = clustering_coefficient_distribution(g, num_bins)
        spectral[i] = spectral_features(g, num_bins)

    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'degrees': degrees,
        'clustering': clustering,
        'spectral': spectral
    }
    return stats


# --------------------------
#  Helper to extract graphs from batched Data
# --------------------------
def extract_individual_graphs(batch_data: Data) -> List[Data]:
    """Extract individual graphs from a batched Data object (device-safe)."""
    if not hasattr(batch_data, "batch"):
        return [batch_data]

    graphs = []
    batch = batch_data.batch
    device = batch_data.x.device if hasattr(batch_data, 'x') else torch.device('cpu')
    num_graphs = int(batch.max().item() + 1)

    for i in range(num_graphs):
        mask = (batch == i)
        node_idx = mask.nonzero(as_tuple=False).view(-1)
        x = batch_data.x[node_idx] if node_idx.numel() > 0 else torch.empty((0, batch_data.x.size(1)), device=device)

        # Extract edges whose endpoints are both in this node set
        if batch_data.edge_index.numel() > 0:
            edge_mask = mask[batch_data.edge_index[0]] & mask[batch_data.edge_index[1]]
            edges = batch_data.edge_index[:, edge_mask]
        else:
            edges = torch.empty((2, 0), dtype=torch.long, device=device)

        # Renumber edges to local node indices
        if edges.size(1) > 0:
            # mapping old idx -> new idx
            mapping = {int(old.item()): int(new) for new, old in enumerate(node_idx)}
            edges = torch.tensor([[mapping[int(e.item())] for e in edges[0]],
                                  [mapping[int(e.item())] for e in edges[1]]],
                                 dtype=torch.long, device=device)
        else:
            edges = torch.empty((2, 0), dtype=torch.long, device=device)

        # y is stored per-graph in batch_data.y 
        y = batch_data.y[i].unsqueeze(0) if hasattr(batch_data, 'y') and batch_data.y is not None and batch_data.y.numel() > i else None

        graphs.append(Data(x=x, edge_index=edges, y=y))

    return graphs



# --------------------------
#  Visualization
# --------------------------
def visualize_graphs(generator: nn.Module, dataset_stats: Tuple[Dict, Dict], num_samples: int, device: torch.device, save_path: str):
    """Visualize a small grid of generated graphs."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    generator.eval()
    with torch.no_grad():
        for i in range(min(num_samples, 6)):
            label = torch.tensor([i % 2], device=device)
            fake = generator(1, dataset_stats, label)

            G = nx.Graph()
            G.add_nodes_from(range(fake.x.size(0)))
            edges = fake.edge_index.cpu().numpy()
            if edges.shape[1] > 0:
                G.add_edges_from(edges.T)

            ax = axes[i]
            colors = 'lightblue' if label.item() == 0 else 'lightcoral'
            nx.draw(G, ax=ax, node_color=colors, node_size=50, with_labels=False)
            ax.set_title(f'Class {label.item()} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.show()



# Uniqueness & novelty
""" def graph_hash(g: Data):
    n = g.x.size(0)
    e = g.edge_index.size(1)
    if e > 0:
        degrees = degree(g.edge_index[0], num_nodes=n).sort()[0]
        deg_seq = ','.join(map(str, degrees.cpu().numpy()[:10]))
    else:
        deg_seq = '0'
    return f"{n}_{e}_{deg_seq}" """

""" def canonicalize_edge_index(edge_index, num_nodes):
    # Sort each edge tuple (u, v)
    edge_index = edge_index.clone()
    # Make edges undirected canonical
    row, col = edge_index
    mask = row > col
    row2 = torch.where(mask, col, row)
    col2 = torch.where(mask, row, col)
    edge_index = torch.stack([row2, col2], dim=0)

    # Lexicographic sort
    perm = (edge_index[0] * num_nodes + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]

    return edge_index """

def canonicalize_edge_index(edge_index):
    """
    Canonical undirected edge_index:
        - (u,v) stored as (min,max)
        - duplicate edges removed
        - lexicographically sorted
    """
    row, col = edge_index

    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    edges = torch.stack([u, v], dim=1)

    # remove duplicate edges
    edges = torch.unique(edges, dim=0)

    # lexicographic sort using stable sorting
    edges = edges[edges[:,1].argsort()]                
    edges = edges[edges[:,0].argsort(stable=True)]    

    return edges.t()

def wl_graph_hash(g, num_iterations=2):
    """
    Pure topological Weisfeiler–Lehman graph hash.
    Structure → hash
    No node features considered.
    """

    # number of nodes inferred from highest index seen
    num_nodes = g.num_nodes

    # Canonicalize edges f
    edge_index = canonicalize_edge_index(g.edge_index)

    # Initial labels = node degree (topology only)
    deg = degree(edge_index[0], num_nodes=num_nodes).tolist()
    labels = [int(d) for d in deg]

    # Build adjacency list
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)
        adj[v].append(u)

    for a in adj:
        a.sort()

    # WL Refinement — purely structural
    for _ in range(num_iterations):
        new_labels = []
        for node in range(num_nodes):
            neigh = sorted(labels[n] for n in adj[node])
            s = (labels[node], *neigh)            
            hashed = hash(s)                      
            new_labels.append(hashed)

        # Renormalize so labels are compact & deterministic
        uniq = {v:i for i,v in enumerate(sorted(set(new_labels)))}
        labels = [uniq[x] for x in new_labels]

    # Graph hash independent of node order
    graph_signature = "_".join(map(str, sorted(labels)))
    return hashlib.md5(graph_signature.encode()).hexdigest()

