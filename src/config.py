from dataclasses import dataclass

# --------------------------
# Configuration
# --------------------------
@dataclass
class Config:
    seed:int = 42
    dataset_name: str = 'PROTEINS' #ENZYMES, MUTAG
    batch_size: int = 64
    train_split: float = 0.7
    val_split: float = 0.15

    noise_dim: int = 16
    class_embed_dim: int = 8
    hidden_dim_gen: int = 32
    hidden_dim_dis: int = 32

    n_critic: int = 5
    lambda_gp: float = 10.0
    epochs: int = 50
    patience: int = 12
    lr_gen: float = 2e-4
    lr_dis: float = 5e-4
    betas_gen: tuple = (0.5, 0.9)
    betas_dis: tuple = (0.5, 0.9)

    start_temperature: float = 2.0
    end_temperature: float = 0.5

    num_samples_per_label: int = 150
    samples_per_batch: int = 50
    num_bins: int = 10
    vis_num_samples: int = 6

    weights = {'degree': 0.3, 'clustering': 0.4, 'spectral': 0.3}

    save_dir: str = './saved_models/'
    results_dir: str = './results/'
    data_dir: str = './data/'
