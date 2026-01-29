import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


# --------------------------
# Edge Predictor
# --------------------------
class EdgePredictor(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, num_edge_types: int = 1):
        super().__init__()
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 4)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 4), num_edge_types)
        )
        self.num_edge_types = num_edge_types

    def forward(self, node_features: torch.Tensor, temperature: float = 1.0):
        device = node_features.device
        n = node_features.size(0)
        h = F.relu(self.node_transform(node_features))

        if n < 2:
            return torch.empty((0, self.num_edge_types), device=device), torch.empty((2, 0), dtype=torch.long, device=device)

        idx_i, idx_j = [], []
        for i in range(n - 1):
            idx_i.append(torch.full((n - i - 1,), i, dtype=torch.long, device=device))
            idx_j.append(torch.arange(i + 1, n, dtype=torch.long, device=device))
        idx_i = torch.cat(idx_i)
        idx_j = torch.cat(idx_j)

        pair_feats = torch.cat([h[idx_i], h[idx_j]], dim=1)
        logits = self.edge_mlp(pair_feats)
        probs = torch.sigmoid(logits / max(1e-6, temperature))
        pair_index = torch.stack([idx_i, idx_j], dim=0)
        return probs, pair_index



# --------------------------
# Alternative Edge Predictor (Simpler): Prefers connecting nodes that are "close" in latent space
# --------------------------
class DistanceEdgePredictor(nn.Module):
    """Predicts edges based on learned distance in latent space."""
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU()
        )
        # Learnable threshold for edge creation
        self.threshold = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, node_features: torch.Tensor, temperature: float = 1.0):
        device = node_features.device
        n = node_features.size(0)
        
        if n < 2:
            return torch.empty((0, 1), device=device), torch.empty((2, 0), dtype=torch.long, device=device)
        
        # Encode nodes
        h = self.node_encoder(node_features)
        
        # Compute pairwise distances
        idx_i, idx_j = [], []
        distances = []
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                idx_i.append(i)
                idx_j.append(j)
                # L2 distance
                dist = torch.norm(h[i] - h[j], p=2)
                distances.append(dist)
        
        if len(distances) == 0:
            return torch.empty((0, 1), device=device), torch.empty((2, 0), dtype=torch.long, device=device)
        
        distances = torch.stack(distances).unsqueeze(1)
        
        # Convert distances to probabilities (closer = higher probability)
        # Using negative distance so smaller distances give higher probabilities
        probs = torch.sigmoid((-distances + self.threshold) / max(temperature, 1e-6))
        
        pair_index = torch.stack([
            torch.tensor(idx_i, device=device),
            torch.tensor(idx_j, device=device)
        ], dim=0)
        
        return probs, pair_index


# --------------------------
# Generator
# --------------------------
class Generator(nn.Module):
    def __init__(self, config, num_classes: int, node_feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.noise_dim = config.noise_dim
        self.class_embedding = nn.Embedding(num_classes, config.class_embed_dim)
        self.node_generator = nn.Sequential(
            nn.Linear(config.noise_dim + config.class_embed_dim, config.hidden_dim_gen),
            nn.ReLU(),
            nn.Linear(config.hidden_dim_gen, node_feat_dim)
        )
        self.edge_predictor = DistanceEdgePredictor(node_feat_dim, config.hidden_dim_gen)

    def sample_node_count(self, class_label: int, node_stats: Dict):
        if node_stats is None or class_label not in node_stats:
            return int(np.random.randint(10, 30))
        stats = node_stats[class_label]
        mean, std = stats['mean'], max(stats['std'], 1.0)
        count = int(np.round(np.random.normal(mean, std * 0.5)))
        return int(np.clip(count, stats['min'], stats['max']))

    def forward(self, num_graphs: int, dataset_stats: Tuple[Dict, Dict], class_labels: torch.Tensor = None, temperature: float = 1.0) -> Data:
        device = next(self.parameters()).device
        if class_labels is None:
            class_labels = torch.randint(0, self.num_classes, (num_graphs,), device=device)
        else:
            class_labels = class_labels.to(device)

        class_embeds = self.class_embedding(class_labels)
        node_stats, edge_stats = dataset_stats

        all_x, all_edges, all_batch = [], [], []
        node_offset = 0

        for i in range(num_graphs):
            label = int(class_labels[i].item())
            num_nodes = self.sample_node_count(label, node_stats)
            z = torch.randn(num_nodes, self.noise_dim, device=device)
            class_expand = class_embeds[i].unsqueeze(0).repeat(num_nodes, 1)
            x = self.node_generator(torch.cat([z, class_expand], dim=1))

            # Edge predictor outputs probabilities for ALL possible edges
            edge_probs, pair_idx = self.edge_predictor(x, temperature)
            m_pairs = pair_idx.size(1) if pair_idx.numel() > 0 else 0


            # Calculate expected density from dataset statistics
            density = 0.05
            if edge_stats and node_stats and label in edge_stats:
                mean_edges = max(0.0, float(edge_stats[label]['mean']))
                mean_nodes = max(1.0, float(node_stats[label]['mean']))
                density = float(np.clip(mean_edges / max((mean_nodes*(mean_nodes-1)/2),1), 0.0, 1.0))

            edges_tensor = torch.empty((2, 0), dtype=torch.long, device=device)
            
            #Select TOP-K edges based on density
            if m_pairs > 0:
                topk = int(round(density * m_pairs))
                if topk > 0:
                    _, idx_topk = torch.topk(edge_probs.view(-1), k=topk, largest=True)
                    sel_pairs = pair_idx[:, idx_topk]
                    u = sel_pairs[0] + node_offset
                    v = sel_pairs[1] + node_offset
                    edges_tensor = torch.cat([torch.stack([u, v], dim=0), torch.stack([v, u], dim=0)], dim=1)

            all_x.append(x)
            all_edges.append(edges_tensor)
            all_batch.append(torch.full((num_nodes,), i, dtype=torch.long, device=device))
            node_offset += num_nodes

        x = torch.cat(all_x, dim=0) if all_x else torch.empty((0, self.node_generator[-1].out_features), device=device)
        edge_index = torch.cat(all_edges, dim=1) if all_edges else torch.empty((2, 0), dtype=torch.long, device=device)
        batch = torch.cat(all_batch, dim=0) if all_batch else torch.empty((0,), dtype=torch.long, device=device)
        y = class_labels.clone().to(device)
        return Data(x=x, edge_index=edge_index, batch=batch, y=y)


# --------------------------
# Discriminator
# --------------------------
class Discriminator(nn.Module):
    def __init__(self, node_feat_dim: int, num_classes: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.class_embedding = nn.Embedding(num_classes, hidden_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: Data, class_labels: torch.Tensor):
        x = F.leaky_relu(self.conv1(data.x, data.edge_index), 0.2)
        x = global_mean_pool(x, data.batch)
        class_embed = self.class_embedding(class_labels.to(x.device))
        x = torch.cat([x, class_embed], dim=1)
        return self.classifier(x)
