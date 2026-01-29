
import os

import logging
from typing import Tuple, Dict,Iterable

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


from utils import extract_individual_graphs, wl_graph_hash, compute_graph_statistics, compute_mmd
# --------------------------
# Gradient Penalty
# --------------------------
def compute_gradient_penalty(discriminator: nn.Module, real_data: Data, fake_data: Data, class_labels: torch.Tensor, device: torch.device, lambda_gp: float = 10.0):
    min_nodes = min(real_data.x.size(0), fake_data.x.size(0))
    if min_nodes == 0:
        return torch.tensor(0.0, device=device)

    alpha = torch.rand(min_nodes, 1, device=device)
    interpolated_x = alpha * real_data.x[:min_nodes] + (1 - alpha) * fake_data.x[:min_nodes]
    interpolated_x.requires_grad_(True)

    edge_mask = (real_data.edge_index[0] < min_nodes) & (real_data.edge_index[1] < min_nodes)
    interpolated_edges = real_data.edge_index[:, edge_mask]
    batch = real_data.batch[:min_nodes]

    interpolated_data = Data(x=interpolated_x, edge_index=interpolated_edges, batch=batch)
    unique_graph_ids = torch.unique(batch)
    adjusted_labels = class_labels[unique_graph_ids.to(class_labels.device)]

    d_interpolated = discriminator(interpolated_data, adjusted_labels)
    grads = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated_x,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gp_per_graph = []
    for gid in unique_graph_ids:
        node_mask = (batch == gid)
        node_grads = grads[node_mask]
        node_norms = node_grads.norm(2, dim=1)
        graph_norm = torch.sqrt((node_norms ** 2).sum())
        gp_per_graph.append((graph_norm - 1.0) ** 2)

    return torch.stack(gp_per_graph).mean() * lambda_gp if gp_per_graph else torch.tensor(0.0, device=device)

# --------------------------
#  Training Functions
# --------------------------
def train_epoch(generator: nn.Module, discriminator: nn.Module, loader: DataLoader, opt_g, opt_d, config, dataset_stats: Tuple[Dict, Dict], device: torch.device, epoch: int = 1):
    """Train for one epoch with temperature annealing (generator & discriminator)."""
    generator.train()
    discriminator.train()

    # Temperature annealing: start random, become more deterministic (the smaller, the more deterministic)
    temperature = max(config.end_temperature, config.start_temperature - (epoch * 0.03))  # 2.0 -> 0.5 over ~50 epochs

    d_losses = []
    g_losses = []

    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.y.size(0) if hasattr(batch, 'y') else 0

        if batch_size == 0:
            continue

        real_labels = batch.y  # shape [num_graphs]

        # Train Discriminator (n_critic times)
        for _ in range(config.n_critic):
            opt_d.zero_grad()

            fake_data = generator(batch_size, dataset_stats, real_labels, temperature=temperature)

            d_real = discriminator(batch, real_labels)
            d_fake = discriminator(fake_data.detach(), real_labels)

            gp = compute_gradient_penalty(discriminator, batch, fake_data, real_labels, device, lambda_gp=config.lambda_gp)

            d_loss = -torch.mean(d_real) + torch.mean(d_fake) + gp
            d_loss.backward()
            opt_d.step()
            d_losses.append(d_loss.item())

        # Train Generator
        opt_g.zero_grad()
        fake_data = generator(batch_size, dataset_stats, real_labels, temperature=temperature)
        d_fake = discriminator(fake_data, real_labels)
        g_loss = -torch.mean(d_fake)
        g_loss.backward()
        opt_g.step()
        g_losses.append(g_loss.item())

    return float(np.mean(d_losses)) if len(d_losses) > 0 else 0.0, float(np.mean(g_losses)) if len(g_losses) > 0 else 0.0


def train(generator: nn.Module, discriminator: nn.Module, train_loader: DataLoader, val_loader: DataLoader, opt_g, opt_d, config, labels, dataset_stats: Tuple[Dict, Dict], device: torch.device, logger: logging.Logger, epoch: int = 1):
    # Training loop
    best_val_score = float('inf')
    patience_counter = 0
    patience = config.patience

    logger.info("Starting training...")
    logger.info("Monitoring: MMD (weighted)")

    for epoch in range(1, config.epochs + 1):
        d_loss, g_loss = train_epoch(generator, discriminator, train_loader,
                                     opt_g, opt_d, config, dataset_stats, device, epoch)

        val_score = evaluate(generator, discriminator, labels, val_loader, train_loader, dataset_stats, config, device, logger)

        logger.info(f"Epoch {epoch:02d} | D_loss: {d_loss:.6f} | G_loss: {g_loss:.6f} | Val: {val_score:.6f}")

        # Early stopping + save
        if val_score < best_val_score:
            best_val_score = val_score
            patience_counter = 0
            save_path = os.path.join(config.save_dir, f'{config.dataset_name}_best_model.pt')
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'epoch': epoch
            }, save_path)
            logger.info(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break



# --------------------------
#  Evaluation 
# --------------------------
def evaluate(generator: nn.Module, discriminator: nn.Module, labels: Iterable[int], eval_loader: DataLoader, train_loader: DataLoader, dataset_stats: Tuple[Dict, Dict], config, device: torch.device, logger: logging.Logger, file: bool = False):
    """
    Comprehensive evaluation using MMD on degree, clustering, and spectral distributions.

    labels: iterable of class ids, e.g. range(num_classes)
    eval_loader: loader for the validation/test set
    train_loader: loader for the train set (used to compute novelty)
    """
    generator.eval()
    discriminator.eval()

    mmd_degree = {}
    mmd_clustering = {}
    mmd_spectral = {}
    mmd_combined = {}
    uniqueness = {}
    novelty = {}

    # Collect real and fake graphs by class
    real_graphs_by_class = {int(l): [] for l in labels}
    fake_graphs_by_class = {int(l): [] for l in labels}

    with torch.no_grad():
        # Collect real graphs from eval_loader and generate fake ones (per-batch)
        for batch in eval_loader:
            batch = batch.to(device)
            real_graphs = extract_individual_graphs(batch)

            # Generate same number of fake graphs with matching labels from the real batch
            fake_batch = generator(batch.y.size(0), dataset_stats, batch.y.to(device))
            fake_graphs = extract_individual_graphs(fake_batch)

            for g in real_graphs:
                if g.y is not None:
                    label = int(g.y.item())
                    if label in real_graphs_by_class:
                        real_graphs_by_class[label].append(g)

            for g in fake_graphs:
                if g.y is not None:
                    label = int(g.y.item())
                    if label in fake_graphs_by_class:
                        fake_graphs_by_class[label].append(g)

    
    real_stats = compute_graph_statistics(real_graphs, num_bins=config.num_bins)
    fake_stats = compute_graph_statistics(fake_graphs, num_bins=config.num_bins)

    mmd_overall_degree = compute_mmd(real_stats['degrees'], fake_stats['degrees'], kernel='rbf', gamma=1.0)
    mmd_overall_clustering = compute_mmd(real_stats['clustering'], fake_stats['clustering'], kernel='rbf', gamma=1.0)
    mmd_overall_spectral = compute_mmd(real_stats['spectral'], fake_stats['spectral'], kernel='rbf', gamma=0.1)

    total_combined = (config.weights['degree'] * mmd_overall_degree +
                                     config.weights['clustering'] * mmd_overall_clustering +
                                     config.weights['spectral'] * mmd_overall_spectral[class_label])

    # ------------------------------
    # Compute MMD for each graph statistic
    # ------------------------------
    

    for class_label in labels:
        real_graphs = real_graphs_by_class.get(int(class_label), [])
        fake_graphs = fake_graphs_by_class.get(int(class_label), [])

        if len(real_graphs) == 0 or len(fake_graphs) == 0:
            logger.info(f"Skipping class {class_label} (real: {len(real_graphs)}, fake: {len(fake_graphs)})")
            continue

        logger.info(f"\n--- Class {class_label} ---")
        logger.info(f"Test graphs: {len(real_graphs)}, Generated graphs: {len(fake_graphs)}")

        real_stats = compute_graph_statistics(real_graphs, num_bins=config.num_bins)
        fake_stats = compute_graph_statistics(fake_graphs, num_bins=config.num_bins)

        logger.info(f"Real - Avg nodes: {np.mean(real_stats['num_nodes']):.1f}±{np.std(real_stats['num_nodes']):.1f}, "
                    f"Avg edges: {np.mean(real_stats['num_edges']):.1f}±{np.std(real_stats['num_edges']):.1f}")
        logger.info(f"Fake - Avg nodes: {np.mean(fake_stats['num_nodes']):.1f}±{np.std(fake_stats['num_nodes']):.1f}, "
                    f"Avg edges: {np.mean(fake_stats['num_edges']):.1f}±{np.std(fake_stats['num_edges']):.1f}")

        # Compute MMDs
        mmd_degree[class_label] = compute_mmd(real_stats['degrees'], fake_stats['degrees'], kernel='rbf', gamma=1.0)
        mmd_clustering[class_label] = compute_mmd(real_stats['clustering'], fake_stats['clustering'], kernel='rbf', gamma=1.0)
        mmd_spectral[class_label] = compute_mmd(real_stats['spectral'], fake_stats['spectral'], kernel='rbf', gamma=0.1)

        logger.info(f"MMD Degree {class_label}: {mmd_degree[class_label]:.6f}")
        logger.info(f"MMD Clustering {class_label}: {mmd_clustering[class_label]:.6f}")
        logger.info(f"MMD Spectral {class_label}: {mmd_spectral[class_label]:.6f}")

        mmd_combined[class_label] = (config.weights['degree'] * mmd_degree[class_label] +
                                     config.weights['clustering'] * mmd_clustering[class_label] +
                                     config.weights['spectral'] * mmd_spectral[class_label])

        
        logger.info(f"MMD Combined {class_label}: {mmd_combined[class_label]:.6f}")
        

        fake_hashes = [wl_graph_hash(g) for g in fake_graphs]
        uni = len(set(fake_hashes)) / len(fake_hashes) if len(fake_hashes) > 0 else 0.0

        # Build some train hashes for novelty 
        train_graphs = []
        for batch in train_loader:
            batch = batch.to(device)
            batch_graphs = extract_individual_graphs(batch)
            for g in batch_graphs:
                if g.y is not None and int(g.y.item()) == int(class_label):
                    train_graphs.append(g)
            #if len(train_graphs) > 100: #(sample up to 100 training graphs of same class)
            #    break

        train_hashes = set([wl_graph_hash(g) for g in train_graphs])
        novel_graphs = [h for h in fake_hashes if h not in train_hashes] #graphs that are generated and that are not identical to training graphs
        novel = len(novel_graphs) / len(fake_hashes) if len(fake_hashes) > 0 else 0.0


        uniqueness[class_label] = uni
        novelty[class_label] = novel
        logger.info(f"Uniqueness: { uniqueness[class_label]:.3f}")
        logger.info(f"Novelty: {novelty[class_label]:.3f}")

    
    logger.info(f"\n--- FINAL RESULTS ---")
    logger.info(f"Combined MMD: {total_combined:.6f}")

    # Save results to file
    if file:
        results_path = os.path.join(config.results_dir, f'{config.dataset_name}_results.txt')
        os.makedirs(config.results_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            f.write("Graph Generation Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"  Overall MMD Degree: {mmd_overall_degree:.6f}\n")
            f.write(f"  Overall MMD Clustering: {mmd_overall_clustering:.6f}\n")
            f.write(f"  Overall MMD Spectral: {mmd_overall_spectral:.6f}\n")
            for class_label in labels:
                real_graphs = real_graphs_by_class.get(int(class_label), [])
                fake_graphs = fake_graphs_by_class.get(int(class_label), [])
                if len(real_graphs) > 0 and len(fake_graphs) > 0:
                    real_stats = compute_graph_statistics(real_graphs, num_bins=config.num_bins)
                    fake_stats = compute_graph_statistics(fake_graphs, num_bins=config.num_bins)
                    f.write(f"Class {class_label}:\n")
                    f.write(f"  Sample size: {len(real_graphs)} real, {len(fake_graphs)} generated\n")
                    f.write(f"  MMD Degree: {mmd_degree[class_label]:.6f}\n")
                    f.write(f"  MMD Clustering: {mmd_clustering[class_label]:.6f}\n")
                    f.write(f"  MMD Spectral: {mmd_spectral[class_label]:.6f}\n")
                    f.write(f"  MMD Combined: {mmd_combined[class_label]:.6f}\n")
                    f.write(f"  Uniqueness: {uniqueness[class_label]:.6f}\n")
                    f.write(f"  Novelty: {novelty[class_label]:.6f}\n")
                    
                    f.write(f"  Avg Nodes: {np.mean(real_stats['num_nodes']):.1f} -> {np.mean(fake_stats['num_nodes']):.1f}\n")
                    f.write(f"  Avg Edges: {np.mean(real_stats['num_edges']):.1f} -> {np.mean(fake_stats['num_edges']):.1f}\n\n")

        logger.info(f"Detailed results saved to: {results_path}")

    return total_combined
