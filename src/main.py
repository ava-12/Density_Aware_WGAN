import os
import random
import logging

import numpy as np
import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from config import Config
from models import Generator, Discriminator
from train import train, evaluate
from utils import visualize_graphs, analyze_dataset_statistics

# --------------------------
# Logging
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --------------------------
#  Main Function
# --------------------------
def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Fixed seeds for reproducibility (optional)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)

    # Setup directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # Load dataset
    dataset = TUDataset(root=config.data_dir, name=config.dataset_name).shuffle()

    # Split dataset
    n = len(dataset)
    train_size = int(config.train_split * n)
    val_size = int(config.val_split * n)

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    logger.info(f"Node features: {dataset.num_node_features}, Classes: {dataset.num_classes}")

    # labels as class ids
    labels = list(range(dataset.num_classes))

    # Analyze statistics (returns node_stats, edge_stats)
    dataset_stats = analyze_dataset_statistics(train_dataset, dataset.num_classes)
    logger.info(f"Node Summary: {dataset_stats[0]}")
    logger.info(f"Edge Summary: {dataset_stats[1]}")
    

    # Initialize models
    generator = Generator(config, dataset.num_classes, dataset.num_node_features).to(device)
    discriminator = Discriminator(dataset.num_node_features, dataset.num_classes, config.hidden_dim_dis).to(device)

    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr_gen, betas=config.betas_gen)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr_dis, betas=config.betas_dis)

    train(generator, discriminator, train_loader, val_loader, opt_g, opt_d, config, labels, dataset_stats, device, logger, config.epochs)


    # Load best model if exists
    save_path = os.path.join(config.save_dir, 'best_model.pt')
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        logger.info(f"Loaded checkpoint from {save_path}, epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        logger.warning("No checkpoint found - using current model weights")

    # Final evaluation
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 50)

    evaluate(generator, discriminator, labels, test_loader, train_loader, dataset_stats, config, device, logger, file=True)

    # Overall summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info("Lower MMD = Better distribution match (< 0.05 excellent, < 0.1 good)")
    logger.info("Higher Uniqueness = More diverse graphs (> 0.9 excellent)")
    logger.info("Higher Novelty = More creative generation (> 0.8 good)")

    # Visualize some samples
    visualize_graphs(generator, dataset_stats, 6, device, os.path.join(config.results_dir, f'{config.dataset_name}_generated_samples.png'))

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
