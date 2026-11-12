#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, pairwise_distances
from sklearn.manifold import MDS
from collections import Counter
import itertools
import math
import argparse

####################
## Hierarchical SOM
####################

## Command line arguments
parser = argparse.ArgumentParser(description="Hierarchical SOM for joint species delimitation.")
parser.add_argument("--input", required=True, help="Input CSV file containing morphology, environment, and dna data.")
parser.add_argument("--outputs", default="outputs", help="Output directory to write plots and CSV outputs.")
parser.add_argument("--iters", type=int, default=500, help="Number of SOM training iterations")
parser.add_argument("--sigma", type=float, default=1.0, help="Initial SOM neighborhood sigma")
parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size for SOM training.")
parser.add_argument("--mds_dim", type=int, default=10, help="Dimension of MDS meta-embedding.")
parser.add_argument("--k_min", type=int, default=2, help="Minimum number of clusters to test in K-Means")
parser.add_argument("--k_max", type=int, default=5, help="Maximum number of clusters to test in K-Means")
parser.add_argument("--seed", type=int, default=1225, help="Random seed")

args = parser.parse_args()

## Settings
print("\n--- Hierarchical SOM Settings ---")
print(f"Input CSV:        {args.input}")
print(f"Outputs Dir:      {args.outputs}")
print(f"Iterations:       {args.iters}")
print(f"Sigma:            {args.sigma}")
print(f"Learning Rate:    {args.lr}")
print(f"Batch Size:       {args.batch_size}")
print(f"MDS Dim:          {args.mds_dim}")
print(f"K Range:          {args.k_min}–{args.k_max-1}")
print(f"Seed:             {args.seed}")
print("---\n")

## Set seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)

## Output directory
os.makedirs(args.outputs, exist_ok=True)

## Hyperparameters
ITERS = args.iters
SIGMA = args.sigma
L_RATE = args.lr
B_SIZE = args.batch_size

## SOM model
class SOM(nn.Module):
    def __init__(self, x, y, input_dim, sigma=1.0, lr=0.7, device='cuda'):
        super(SOM, self).__init__()
        self.x = x
        self.y = y
        self.input_dim = input_dim
        self.sigma = sigma
        self.lr = lr
        self.device = device
        self.weights = torch.randn(x * y, input_dim, device=device)
        xx, yy = torch.meshgrid(torch.arange(x), torch.arange(y), indexing='ij')
        self.coords = torch.stack([xx.flatten(), yy.flatten()], dim=1).float().to(device)

    def forward(self, x):
        dists = torch.cdist(x, self.weights)
        bmu_idx = torch.argmin(dists, dim=1)
        return bmu_idx

    def update(self, x, bmu_idx, iter_count, max_iter):
        lr = self.lr * (1 - iter_count / max_iter)
        sigma = self.sigma * (1 - iter_count / max_iter)
        bmu_coords = self.coords[bmu_idx]
        dists = torch.cdist(bmu_coords, self.coords)
        h = torch.exp(-dists**2 / (2 * sigma**2)).unsqueeze(2)
        delta = (x.unsqueeze(1) - self.weights.unsqueeze(0)) * h
        self.weights += lr * delta.sum(dim=0)

    def quantization_error(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dists = torch.cdist(X_t, self.weights)
            min_dists, _ = torch.min(dists, dim=1)
        return min_dists.mean().item()

    def train_som(self, data, num_iters=1000, batch_size=None, track_qe=False):
        if batch_size is None:
            batch_size = min(8, data.shape[0])
        data_t = torch.tensor(data, dtype=torch.float32, device=self.device)
        n_samples = len(data_t)
        
        qe_history = []

        for it in range(num_iters):
            idx = torch.randperm(n_samples, device=self.device)[:batch_size]
            batch = data_t[idx]
            bmu_idx = self.forward(batch)
            self.update(batch, bmu_idx, it, num_iters)

            if track_qe:
                qe = self.quantization_error(data)
                qe_history.append(qe)
        
        if track_qe:
            return qe_history

    def get_weights(self):
        return self.weights.detach().cpu().numpy()

    def winner_batch(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dists = torch.cdist(X_t, self.weights)
            bmu_idx = torch.argmin(dists, dim=1)
            coords = self.coords[bmu_idx].cpu().numpy().astype(int)
        return coords
    
## Load data
print("Loading dataset...")
df = pd.read_csv(args.input)
morph_cols = [c for c in df.columns if c.startswith('morph_')]
env_cols = [c for c in df.columns if c.startswith('env_')]
dna_cols = [c for c in df.columns if c.startswith('dna_')]

feature_groups = {}
if morph_cols: feature_groups['Morph'] = morph_cols
if env_cols: feature_groups['Env'] = env_cols
if dna_cols: feature_groups['DNA'] = dna_cols

ids = df['ID'].astype(str).values
print(f"Data types detected: {list(feature_groups.keys())}, {len(ids)} samples")

## SOM grid (heuristic)
n_samples = len(df)
g = 5 * math.sqrt(n_samples)
som_size = int(round(math.sqrt(g)))
som_x, som_y = som_size, som_size
print(f"SOM grid size: {som_x}x{som_y} (g={g:.2f}, n={n_samples})")

## Data type combinations
group_names = list(feature_groups.keys())
all_combinations = []
for r in range(1, len(group_names) + 1):
    for combo in itertools.combinations(group_names, r):
        all_combinations.append(combo)
print(f"Total data type combinations: {len(all_combinations)}")

## Normalize each data combination
print("Normalizing data types...")
group_scaled = {}
for combo in all_combinations:
    combo_name = '+'.join(combo)
    X_combo = np.concatenate([df[feature_groups[g]].values for g in combo], axis=1)
    scaler = MinMaxScaler()
    group_scaled[combo_name] = scaler.fit_transform(X_combo)
    print(f"  {combo_name}: {X_combo.shape[1]} features normalized.")

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device:", device)

## K-means 
def auto_kmeans_colors(weights, k_min=2, k_max=5, colormap='tab10'):
    best_score = -1
    best_k = k_min
    k_max = min(k_max, len(weights))
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, n_init=10).fit(weights)
        if len(np.unique(kmeans.labels_)) <= 1:
            continue
        score = silhouette_score(weights, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k
    kmeans = KMeans(n_clusters=best_k, n_init=10).fit(weights)
    labels = kmeans.labels_
    cmap = plt.get_cmap(colormap)
    cluster_colors = {i: cmap(i / best_k) for i in range(best_k)}
    return labels, best_k, cluster_colors

## SOM Layer-1
layer1_soms = {}
layer1_bmu_coords = {}      
layer1_latent_by_sample = {}
layer1_neuron_to_cluster = {} 
layer1_codebooks = {}       
layer1_mappings = {}        
layer1_outputs = {}

print("\n--- Training SOM Layer-1 ---")
for combo_name, X_scaled in group_scaled.items():
    print(f"Processing Layer-1 combo: {combo_name} (shape {X_scaled.shape})")
    input_dim = X_scaled.shape[1]
    som = SOM(som_x, som_y, input_dim, sigma=SIGMA, lr=L_RATE, device=device)
    batch_size = B_SIZE if X_scaled.shape[0] >= B_SIZE else None
    
    ## Train with QE history
    qe_history = som.train_som(X_scaled, num_iters=ITERS, batch_size=batch_size, track_qe=True)

    ## Write QE history
    qe_df = pd.DataFrame({'iteration': np.arange(1, ITERS+1), 'quantization_error': qe_history})
    qe_df.to_csv(f"{args.outputs}/layer1_qe_history_{combo_name}.csv", index=False)
    print(f"  QE history for Layer-1 combo writtent to {combo_name}")

    layer1_soms[combo_name] = som

    coords = som.winner_batch(X_scaled)
    layer1_bmu_coords[combo_name] = coords

    codebook = som.get_weights().reshape(-1, input_dim)
    layer1_codebooks[combo_name] = codebook

    flattened_bmu_idx = coords[:,0] * som_y + coords[:,1]
    latent_vectors = codebook[flattened_bmu_idx]
    layer1_latent_by_sample[combo_name] = latent_vectors
    layer1_outputs[combo_name] = latent_vectors 

    labels, best_k, cluster_colors = auto_kmeans_colors(codebook, k_min=args.k_min, k_max=args.k_max)
    neuron_to_cluster = {}
    for idx in range(len(labels)):
        i = idx // som_y
        j = idx % som_y
        neuron_to_cluster[(i,j)] = labels[idx]
    layer1_neuron_to_cluster[combo_name] = neuron_to_cluster

    mapping = {}
    for sample_idx, (i,j) in enumerate(coords):
        mapping.setdefault((i,j), []).append(sample_idx)
    layer1_mappings[combo_name] = mapping

    pd.DataFrame(latent_vectors, columns=[f"somfeat_{i}" for i in range(latent_vectors.shape[1])]) \
        .to_csv(f"{args.outputs}/layer1_output_{combo_name}.csv", index=False)
    pd.DataFrame(codebook, columns=[f"codebook_{i}" for i in range(codebook.shape[1])]) \
        .to_csv(f"{args.outputs}/layer1_codebook_{combo_name}.csv", index=False)

    print(f"  Trained SOM for {combo_name}: SOM size {som_x}x{som_y}, input_dim={input_dim}, neuron_k={best_k}")

print("\nFinished training all Layer-1 SOMs and collecting outputs.")

## Distance Fusion to MDS
def compute_modality_distance_from_latent(latent_X):
    D = pairwise_distances(latent_X, metric="euclidean")
    n = D.shape[0]
    if n <= 1:
        return D
    mask = ~np.eye(n, dtype=bool)
    mean_d = D[mask].mean()
    if mean_d == 0:
        return D
    return D / mean_d

print("\n--- META-LEVEL: compute distance matrices for each Layer-1 output ---")
per_combo_distances = {}
for combo_name, latent in layer1_outputs.items():
    D = compute_modality_distance_from_latent(latent)
    per_combo_distances[combo_name] = D
    pd.DataFrame(D).to_csv(f"{args.outputs}/dist_matrix_layer1_{combo_name}.csv", index=False)
    print(f"  Saved distance matrix for {combo_name} (shape {D.shape})")

print("\nFusing distances across Layer-1 outputs (mean fusion)...")
all_Ds = np.stack(list(per_combo_distances.values()), axis=2) 
fused_D = np.mean(all_Ds, axis=2)
pd.DataFrame(fused_D).to_csv(f"{args.outputs}/dist_matrix_fused_mean.csv", index=False)
print(f"Fused distance matrix written to {args.outputs}/dist_matrix_fused_mean.csv")

mds_dim = args.mds_dim
print(f"\nApplying MDS once on fused distance matrix to {mds_dim}D meta-embedding...")
mds = MDS(n_components=mds_dim, dissimilarity='precomputed', random_state=args.seed)
meta_embedding = mds.fit_transform(fused_D)
print("Meta-embedding shape:", meta_embedding.shape)
pd.DataFrame(meta_embedding, columns=[f"mds_{i}" for i in range(meta_embedding.shape[1])]) \
    .to_csv(f"{args.outputs}/meta_embedding_mds{mds_dim}.csv", index=False)

print("\n--- Training Layer-2 (Meta SOM) on MDS meta-embedding ---")
som2 = SOM(som_x, som_y, meta_embedding.shape[1], sigma=SIGMA, lr=L_RATE, device=device)

## Train with QE history
qe_history_meta = som2.train_som(meta_embedding, num_iters=ITERS, batch_size=B_SIZE, track_qe=True)

## Write QE history
qe_meta_df = pd.DataFrame({'iteration': np.arange(1, ITERS+1), 'quantization_error': qe_history_meta})
qe_meta_df.to_csv(f"{args.outputs}/layer2_qe_history_meta_som.csv", index=False)
print("  QE history saved for Layer-2 Meta SOM")

## SOM Layer-2 (Meta-SOM)
print("\n--- Training Layer-2 (Meta SOM) on MDS meta-embedding ---")
som2 = SOM(som_x, som_y, meta_embedding.shape[1], sigma=SIGMA, lr=L_RATE, device=device)
som2.train_som(meta_embedding, num_iters=ITERS, batch_size=B_SIZE)

coords_layer2 = som2.winner_batch(meta_embedding)
mapping_layer2 = {}
for idx, (i,j) in enumerate(coords_layer2):
    mapping_layer2.setdefault((i,j), []).append(idx)

weights2 = som2.get_weights().reshape(-1, meta_embedding.shape[1])
labels2, best_k_layer2, cluster_colors_layer2 = auto_kmeans_colors(weights2, k_min=args.k_min, k_max=args.k_max)
neuron_to_cluster_layer2 = {(i,j): labels2[i*som_y+j] for i in range(som_x) for j in range(som_y)}

meta_labels = np.zeros(len(ids), dtype=int)
for idx, (i,j) in enumerate(coords_layer2):
    meta_labels[idx] = neuron_to_cluster_layer2[(i,j)]

pd.DataFrame(coords_layer2, columns=['som_x', 'som_y']).assign(ID=ids).to_csv(f"{args.outputs}/meta_som_coords.csv", index=False)
pd.DataFrame({'ID': ids, 'meta_cluster': meta_labels}).to_csv(f"{args.outputs}/meta_som_labels.csv", index=False)
pd.DataFrame(weights2, columns=[f"meta_codebook_{i}" for i in range(weights2.shape[1])]).to_csv(f"{args.outputs}/meta_codebook.csv", index=False)

print("\nFinished training Meta-SOM.")

## Ablation
print("\n--- Comparing each Layer-1 ARI and NMI to Meta-SOM ---")
comparison_results = []
for combo_name, neuron_to_cluster in layer1_neuron_to_cluster.items():
    coords = layer1_bmu_coords[combo_name]
    per_sample_labels = np.zeros(len(ids), dtype=int)
    for sidx, (i,j) in enumerate(coords):
        per_sample_labels[sidx] = neuron_to_cluster.get((i,j), -1)

    ari_val = adjusted_rand_score(meta_labels, per_sample_labels)
    nmi_val = normalized_mutual_info_score(meta_labels, per_sample_labels)
    comparison_results.append({'combination': combo_name, 'ari_vs_meta': ari_val, 'nmi_vs_meta': nmi_val})
    print(f"  {combo_name}: ARI={ari_val:.4f}, NMI={nmi_val:.4f}")

comparison_df = pd.DataFrame(comparison_results).sort_values(by='ari_vs_meta', ascending=False)
comparison_df.to_csv(f"{args.outputs}/ari_nmi_vs_meta.csv", index=False)
print(f"ARI and NMI summary written to {args.outputs}/ari_nmi_vs_meta.csv")

## ARI and NMI bar charts 
def save_ari_nmi_barchart(df, outpath):
    labels = df['combination'].tolist()
    ari_vals = df['ari_vs_meta'].tolist()
    nmi_vals = df['nmi_vs_meta'].tolist()
    x = np.arange(len(labels))
    width = 0.6
    color_blue = "#4A90E2"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.bar(x, ari_vals, width, edgecolor='black', color=color_blue)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title("Layer-1 ARI vs Meta SOM")
    ax.set_ylabel("Adjusted Rand Index")
    for i, v in enumerate(ari_vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    ax = axes[1]
    ax.bar(x, nmi_vals, width, edgecolor='black', color=color_blue)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title("Layer-1 NMI vs Meta SOM")
    ax.set_ylabel("Normalized Mutual Information")
    for i, v in enumerate(nmi_vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"ARI and NMI bar charts written to {outpath}")

save_ari_nmi_barchart(comparison_df, f"{args.outputs}/ari_nmi_vs_meta.pdf")

## Plot QE convergence for Layer-2 Meta SOM
plt.figure(figsize=(8,6))
plt.plot(np.arange(1, ITERS+1), qe_history_meta, color='#4A90E2', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Quantization Error (QE)")
plt.title("Layer-2 Meta SOM QE Convergence")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"{args.outputs}/meta-SOM_qe_convergence.pdf", bbox_inches='tight')
plt.close()
print(f"  QE convergence plot written as {args.outputs}/meta-SOM_qe_convergence.pdf")

## Plot SOM tiles
def hexagon(cx, cy, size=1.0):
    angles = np.linspace(0, 2*np.pi, 7)
    return cx + size*np.cos(angles), cy + size*np.sin(angles)

marker_list = ['o','s','D','^','v','P','*','X','H','d']
id_to_marker = {uid: marker_list[i % len(marker_list)] for i, uid in enumerate(np.unique(ids))}
empty_color = '#e0e0e0'

all_soms = list(layer1_soms.items()) + [('Meta', som2)]
n_plots = len(all_soms)
n_cols = 3
n_rows = int(np.ceil(n_plots / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
axes = axes.flatten()

for ax_idx, (combo_name, som) in enumerate(all_soms):
    ax = axes[ax_idx]
    legend_handles = []

    if combo_name != 'Meta':
        mapping = layer1_mappings[combo_name]
        neuron_to_cluster = layer1_neuron_to_cluster[combo_name]
    else:
        mapping = mapping_layer2
        neuron_to_cluster = neuron_to_cluster_layer2

    used_clusters = sorted({neuron_to_cluster[(i,j)] for (i,j) in mapping if (i,j) in neuron_to_cluster})
    best_k = len(used_clusters)
    cluster_color_dict = {cluster_id: plt.get_cmap('tab10')(i / max(1,best_k)) 
                          for i, cluster_id in enumerate(used_clusters)}

    for i in range(som_x):
        for j in range(som_y):
            x_offset = i * np.sqrt(3)/2
            y_offset = j + (0.5 if i % 2 else 0)
            if (i,j) in mapping and len(mapping[(i,j)]) > 0:
                cluster_id = neuron_to_cluster[(i,j)]
                color = cluster_color_dict.get(cluster_id, '#000000')
                ids_here = [ids[idx] for idx in mapping[(i,j)]]
                dominant_id = Counter(ids_here).most_common(1)[0][0]
                marker = id_to_marker[dominant_id]

                ax.plot(x_offset, y_offset, marker=marker, markersize=12,
                        markerfacecolor='none', markeredgecolor='white', markeredgewidth=2)

                ax.text(x_offset, y_offset-0.3, str(len(mapping[(i,j)])),
                        ha='center', va='center', fontsize=8, color='white')
            else:
                color = empty_color

            x_hex, y_hex = hexagon(x_offset, y_offset, size=0.5)
            ax.fill(x_hex, y_hex, facecolor=color, edgecolor='gray', linewidth=0.6)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{combo_name} SOM (K={best_k})", fontsize=14)

    for cluster_id, color in cluster_color_dict.items():
        legend_handles.append(ax.plot([], [], 's', color=color, label=f'Cluster {cluster_id}', markersize=10)[0])

    for uid, marker in id_to_marker.items():
        legend_handles.append(ax.plot([], [], marker=marker, color='white',
                                      markerfacecolor='none', markeredgecolor='black',
                                      markersize=10, label=f'ID {uid}')[0])

    legend_handles.append(ax.plot([], [], 's', color=empty_color, label='Empty', markersize=10)[0])

    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5),
              frameon=False, fontsize=8, ncol=2)

for ax_idx in range(len(all_soms), n_rows*n_cols):
    axes[ax_idx].axis('off')

plt.tight_layout()
plt.savefig(f"{args.outputs}/som_tiles.pdf", bbox_inches='tight')
plt.close()
print(f"SOM tile plot written to '{args.outputs}/som_tiles.pdf'")

## Write CSV outputs
for combo_name, X_scaled in group_scaled.items():
    pd.DataFrame(X_scaled, columns=[f"feat_{i}" for i in range(X_scaled.shape[1])]) \
        .to_csv(f"{args.outputs}/normalized_{combo_name}.csv", index=False)

for combo_name, coords in layer1_bmu_coords.items():
    pd.DataFrame(coords, columns=['bmu_x','bmu_y']).assign(ID=ids).to_csv(f"{args.outputs}/layer1_bmu_coords_{combo_name}.csv", index=False)
    neuron_map = layer1_neuron_to_cluster[combo_name]
    neuron_map_df = pd.DataFrame([{'i':i,'j':j,'cluster':c} for (i,j),c in neuron_map.items()])
    neuron_map_df.to_csv(f"{args.outputs}/layer1_neuron_clusters_{combo_name}.csv", index=False)

print(f"All outputs saved successfully in '{args.outputs}/'")
