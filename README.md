# Density-Aware Graph Generation with WGANs 

Graph generation featuring density-aware edge prediction.

## Key Features

- Density-Aware Edge Generation: Respects the edge density distribution of real graphs
- Class-Conditional Generation: Generate graphs with specific structural properties
- Comprehensive Evaluation: MMD metrics for degree, clustering, and spectral features
- Novelty Detection: Tracks uniqueness and novelty of generated graphs


## Model Architecture

### 1. **Generator**
- Transforms noise vectors into node features
- Samples appropriate number of nodes per class
- Uses class embeddings for conditional generation

### 2. **Edge Predictor**
- Computes edges based on latent space proximity
- Faster and more interpretable
- Ideal for spatial graphs

### 3. **Discriminator**
- Graph Convolutional Network (GCN) for processing graphs
- Global mean pooling for graph-level representation
- Class-conditional scoring

<img width="1756" height="651" alt="wgan_edge_graph_architectureN" src="https://github.com/user-attachments/assets/ddb79dc8-b3b6-424a-9bec-c6a70a0aa543" />



## Dataset (TUDatasets)

| Dataset | Nodes | Classes | Description |
|---------|-------|---------|-------------|
| PROTEINS | 20-600 | 2 | Protein structures |
| MUTAG | 10-30 | 2 | Molecular graphs |
| ENZYMES | 10-125 | 6 | Protein tertiary structures |

## Training

- Early Stopping: Automatically stops when validation MMD plateaus
- Model Checkpointing: Saves best model based on validation metrics
- Temperature Annealing: Gradually decreases from 2.0 to 0.5
- Critic Training: Updates discriminator `n_critic` times per generator update


##  Evaluation

### Metrics

The model is evaluated using **Maximum Mean Discrepancy (MMD)** across three graph statistics:

1. **Degree Distribution** - Node connectivity patterns
2. **Clustering Coefficient** - Local graph density
3. **Spectral Features** - Global graph structure (eigenvalues)

### Combined Score

```python
MMD_total = 0.4 * MMD_degree + 0.4 * MMD_clustering + 0.2 * MMD_spectral
```

### Additional Metrics

- **Uniqueness**: Percentage of unique generated graphs
- **Novelty**: Percentage of graphs not seen during training


## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `noise_dim` | 16 | Dimension of noise vector |
| `hidden_dim_gen` | 32 | Generator hidden dimension |
| `hidden_dim_dis` | 32 | Discriminator hidden dimension |
| `n_critic` | 5 | Discriminator updates per generator update |
| `lambda_gp` | 10.0 | Gradient penalty coefficient |
| `lr_gen` | 2e-4 | Generator learning rate |
| `lr_dis` | 5e-4 | Discriminator learning rate |
| `start_temperature` | 2.0 | Initial temperature for edge sampling |
| `end_temperature` | 0.5 | Final temperature for edge sampling |
| `epochs` | 50 | Training epochs |
| `patience` | 12 | Early stopping patience |



## Results


Table 1: Detailed experimental results across all datasets and classes
| Dataset  | Class   | MMD Degree | MMD Clustering | MMD Spectral | Avg Nodes (Real→Gen) | Avg Edges (Real→Gen) | MMD Combined | Uniqueness | Novelty |
| -------- | ------- | ---------- | -------------- | ------------ | -------------------- | -------------------- | ------------ | ---------- | ------- |
| MUTAG    | Class 0 | 0.249      | 1.053          | 0.102        | 13.4→13.4            | 14.0→13.7            | 0.527        | 0.857      | 1.000   |
| MUTAG    | Class 1 | 0.266      | 1.059          | 0.107        | 20.7→19.7            | 23.5→22.1           | 0.527        | 0.933      | 0.933   |
| ENZYMES  | Class 0 |0.206      | 0.200         | 0.133        | 31.6→41.0            | 63.3→41.0            | 0.182        | 1.000      | 1.000   |
| ENZYMES  | Class 1 | 0.137      | 0.127          | 0.059        | 32.1→31.6            | 62.1→68.9            | 0.110        | 1.000      | 0.941   |
| ENZYMES  | Class 2 | 0.151      | 0.135          | 0.048        | 30.1→30.5            | 60.3→62.2            | 0.122        | 1.000      | 0.933   |
| ENZYMES  | Class 3 | 0.208      | 0.165          | 0.044        | 37.4→37.1            | 73.5→71.2            | 0.141        | 1.000      | 0.833   |
| ENZYMES  | Class 4 | 0.148      | 0.114          | 0.114        | 32.1→28.4            | 60.4→53.0            | 0.106        | 1.000      | 1.000   |
| ENZYMES  | Class 5 | 0.149      | 0.121          | 0.042        | 38.8→27.7            | 77.3→54.0            | 0.106        | 0.941      | 0.882  |
| PROTEINS | Class 0 | 0.054      | 0.039          | 0.192        | 54.5→49.5            | 103.0→124.9          | 0.089        | 0.951      | 1.000   |
| PROTEINS | Class 1 | 0.091      | 0.050          | 0.062        | 19.6→25.1            | 36.5→57.2            | 0.066        | 0.921      | 0.859   |


Table 2: Comparison with state-of-the-art methods using MMD metrics (lower is better)
| Model            | PROTEINS (Degree) | PROTEINS (Clustering) | PROTEINS (Spectral)   | ENZYMES (Degree) | ENZYMES (Clustering) |ENZYMES (Spectral) |
| ---------------- | ----------------- | --------------------- | --------------------- | ---------------- | -------------------- | -------------------- |
| DeepGMG          | 0.96              | 0.63                  | -                   | 0.43             | 0.38                 |-                 |
| GraphRNN         | 0.04              | 0.18                  | -                   | 0.06             | 0.20                 |-                |
| LGGAN            | 0.18              | 0.15                  | -                   | 0.09             | 0.17                 |-               |
| WPGAN            | **0.03**          | 0.31                  | -                   | **0.02**         | 0.28                 |-              |
| **Our Approach** | 0.08              | **0.07**              | 0.06                  | 0.09             | **0.08**             |0.04               |

*Lower MMD scores indicate better match to real distribution*

## Distribution Comparison
![PROTEINS_Clustering](https://github.com/user-attachments/assets/771630f9-6da5-4b94-8153-c8ff3c177a90)
Clustering coefficient distribution demonstrates strong alignment with substantial overlap, confirming the model’s superior preservation of local connectivity patterns and community structure, consistent with the best clustering MMD (0.07) achieved among compared methods.


![PROTEINS_Degree](https://github.com/user-attachments/assets/862a3593-60db-49ae-85e1-a69ae6fbfe6f)

The real graphs follow a broader, near-normal degree distribution, reflecting
substantial variation in node connectivity. However, the generated graphs display
a narrower distribution across bins. This behavior highlights the effect of deterministic density-aware edge selection, which constrains graphs toward class-level average sparsity and limits the model’s ability to capture degree heterogeneity.



## Visual Comparison on Real and Generated dataset


<table align="center">
  <tr>
    <td align="center"><b>Class 0 (Real)</b></td>
    <td align="center"><b>Class 1 (Real)</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/76acb433-1caa-4257-be9e-95d75dc3ffae" width="350"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/0500e5ae-8e5a-4bfd-91b4-5412f42a14f7" width="350"/>
    </td>
  </tr>
  <tr>
    <td align="center"><b>Class 0 (Generated)</b></td>
    <td align="center"><b>Class 1 (Generated)</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/90fad3a1-233a-40d0-8267-ecc89b7e96b1" width="350"/>
    </td>
          <img src="https://github.com/user-attachments/assets/810cbacb-bde4-46fa-84a9-beaf08edbc85" width="350"/>
    </td>
  </tr>
</table>




## Citation
Seyedeh Ava Razi Razavi, James Sargant, Sheridan Houghten, and
Renata Dividino. Density-aware graph generation with learnable edge
prediction. In 39th Canadian Conference on Artificial Intelligence (To
Appear). Canadian Artificial Intelligence Association (CAIAC), 2026.
