#!/usr/bin/env python3

import os
import argparse
import itertools
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import random

####################
## iDEC Single
####################

## Command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="iDEC clustering (single modal) for joint species delimitation" )
    parser.add_argument("--input", "-i", required=True, help="Input CSV containing morphology, environment, and dna data.")
    parser.add_argument("--plots_dir", default="plots", help="Directory to write plots.")
    parser.add_argument("--outputs_dir", default="outputs", help="Directory to write latent space, cluster assingments, and other outputs.")

    ## Hyperparameters
    parser.add_argument("--latent_dim", type=int, default=10, help="Latent dimension.")
    parser.add_argument("--n_iter", type=int, default=50, help="Number of iDEC training iterations.")
    parser.add_argument("--update_interval", type=int, default=10, help="Update interval for cluster centers.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter for soft assignment.")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAMMA clustering loss weight.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")

    ## K selection to initialize iDEC
    parser.add_argument("--k_min", type=int, default=2, help="Minimum K to test.")
    parser.add_argument("--k_max", type=int, default=5, help="Maximum K to test.")

    ## Plot legend wrapping
    parser.add_argument("--wrap_width", type=int, default=30, help="Line-wrap width for legend labels.")
    parser.add_argument("--seed", type=int, default=1215, help="Random seed for reproducibility.")

    return parser.parse_args()

args = parse_args()

## Apply arguments (xtian feels really lazy today to edit this redundant section :) haha)
PLOTS_DIR = args.plots_dir
OUTPUTS_DIR = args.outputs_dir
CSV_PATH = args.input

LATENT_DIM = args.latent_dim
N_ITER = args.n_iter
UPDATE_INTERVAL = args.update_interval
ALPHA = args.alpha
GAMMA = args.gamma
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size

K_RANGE = range(args.k_min, args.k_max)
WRAP_WIDTH = args.wrap_width
SEED = args.seed

## Seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("\n--- iDEC Single Modal Settings ---")
print(f"Input CSV:        {CSV_PATH}")
print(f"Plots Dir:        {PLOTS_DIR}")
print(f"Outputs Dir:      {OUTPUTS_DIR}")
print(f"Latent Dim:       {LATENT_DIM}")
print(f"Iterations:       {N_ITER}")
print(f"Update Interval:  {UPDATE_INTERVAL}")
print(f"Alpha:            {ALPHA}")
print(f"Gamma:            {GAMMA}")
print(f"LR:               {LEARNING_RATE}")
print(f"Batch Size:       {BATCH_SIZE}")
print(f"K Range:          {args.k_min}–{args.k_max-1}")
print(f"Wrap Width:       {WRAP_WIDTH}")
print(f"Seed:             {SEED}")
print("---\n")

## iDEC functions
def soft_assignment(z, centers, alpha=1.0):
    dist = np.sum((z[:, np.newaxis, :] - centers)**2, axis=2)
    q = 1.0 / (1.0 + dist/alpha)
    q = q ** ((alpha+1.0)/2.0)
    q = q / np.sum(q, axis=1, keepdims=True)
    return q.astype(np.float32)

def target_distribution(q):
    weight = q ** 2 / q.sum(axis=0)
    return (weight.T / weight.sum(axis=1)).T.astype(np.float32)

def plot_embedding(latent, y_pred, ids, tag):
    """
    Write PCA and t-SNE plots with ellipses and other outputs.
    """
    ## Compute embeddings
    pca_emb = PCA(n_components=2).fit_transform(latent)
    tsne_emb = TSNE(n_components=2, random_state=SEED).fit_transform(latent)

    ## Write PCA CSV
    pca_df = pd.DataFrame(pca_emb, columns=["PC1", "PC2"])
    pca_df.insert(0, "ID", ids)
    pca_df["cluster"] = y_pred
    pca_csv_path = os.path.join(OUTPUTS_DIR, f"{tag}_pca_embeddings.csv")
    pca_df.to_csv(pca_csv_path, index=False)
    print(f"PCA embeddings written to {pca_csv_path}")

    # Write t-SNE CSV
    tsne_df = pd.DataFrame(tsne_emb, columns=["TSNE1", "TSNE2"])
    tsne_df.insert(0, "ID", ids)
    tsne_df["cluster"] = y_pred
    tsne_csv_path = os.path.join(OUTPUTS_DIR, f"{tag}_tsne_embeddings.csv")
    tsne_df.to_csv(tsne_csv_path, index=False)
    print(f"t-SNE embeddings written to {tsne_csv_path}")

    def _make_plot(emb, name):
        ## Set colors
        unique_clusters = np.unique(y_pred)
        n_clusters = len(unique_clusters)
        if n_clusters <= 10:
            cmap = matplotlib.colormaps['tab10']
        else:
            cmap = matplotlib.colormaps['tab20']
        cluster_colors = [cmap(i / max(n_clusters-1, 1)) for i in range(n_clusters)]
        cluster_color_map = {cid: cluster_colors[i] for i, cid in enumerate(unique_clusters)}

        fig, ax = plt.subplots(figsize=(10, 8))
        ## Eoints
        for i, (x, y) in enumerate(emb):
            ax.scatter(x, y, color=cluster_color_map[y_pred[i]], s=50, alpha=0.7)
        ## Ellipses
        for cid in unique_clusters:
            points = emb[y_pred == cid]
            if len(points) > 1:
                mean = points.mean(axis=0)
                cov = np.cov(points.T)
                vals, vecs = np.linalg.eigh(cov)
                width, height = 2 * np.sqrt(np.clip(vals, 1e-8, None))
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                                  alpha=0.2, color=cluster_color_map[cid])
                ax.add_patch(ellipse)

        legend_elements = [
            Patch(color=cluster_color_map[cid],
                  label="\n".join(textwrap.wrap(
                      f"Cluster {cid}: " + ", ".join(ids[y_pred == cid]), width=WRAP_WIDTH)))
            for cid in unique_clusters
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=False)
        ax.set_title(f"{name.upper()} embedding ({tag})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(False)
        plt.tight_layout()
        out_pdf = os.path.join(PLOTS_DIR, f"{tag}_{name}.pdf")
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        print(f"written plot {out_pdf}")

    _make_plot(pca_emb, "pca")
    _make_plot(tsne_emb, "tsne")

def plot_soft_cluster_assignments(q, y_pred, ids, tag):
    """
    Soft cluster assignments
    """
    n_samples, n_clusters = q.shape

    ## Write soft cluster assignment to CSV
    q_df = pd.DataFrame(q, columns=[f"cluster_{k}" for k in range(n_clusters)])
    q_df.insert(0, "ID", ids)
    q_csv_path = os.path.join(OUTPUTS_DIR, f"{tag}_soft_assignments.csv")
    q_df.to_csv(q_csv_path, index=False)
    print(f"Soft assignments written to {q_csv_path}")

    ## Sorted plot
    sort_idx = np.argsort(y_pred)
    q_sorted = q[sort_idx]
    ids_sorted = ids[sort_idx]

    cmap = matplotlib.colormaps['tab20']
    cluster_colors = [cmap(i % 20) for i in range(n_clusters)]
    fig, ax = plt.subplots(figsize=(max(12, n_samples / 5), 6))
    for i in range(n_samples):
        probs = q_sorted[i]
        sorted_idx_prob = np.argsort(-probs)
        bottom = 0
        for k in sorted_idx_prob:
            ax.bar(i, probs[k], bottom=bottom, color=cluster_colors[k], width=0.8)
            bottom += probs[k]

    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(ids_sorted, rotation=90, fontsize=6)
    ax.set_xlabel("Samples (sorted by cluster)")
    ax.set_ylabel("Soft Assignment Probability")
    ax.set_title(f"Soft Cluster Assignments ({tag})")
    legend_handles = [Patch(color=cluster_colors[k], label=f"Cluster {k}") for k in range(n_clusters)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(False)
    plt.tight_layout()
    out_pdf = os.path.join(PLOTS_DIR, f"soft_cluster_assignments_{tag}.pdf")
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"Written plot {out_pdf}")

def plot_ari_nmi_barchart(summary_df, outpath):
    """
    Plot ARI and NMI bar charts
    """
    labels = summary_df['ablation'].tolist()
    ari_vals = summary_df['ari_vs_full'].tolist()
    nmi_vals = summary_df['nmi_vs_full'].tolist()
    x = np.arange(len(labels))
    width = 0.6
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, vals, title, ylabel in zip(
            axes, [ari_vals, nmi_vals], ['ARI', 'NMI'],
            ['Adjusted Rand Index', 'Normalized Mutual Info']):
        ax.bar(x, vals, width, edgecolor='black')
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title(f"{title} vs Full")
        ax.set_ylabel(ylabel)
        ax.grid(False)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"Written ARI and NMI bar charts to {outpath}")

## Build autoencoder
def build_autoencoder(input_dim, latent_dim=LATENT_DIM, dropout_rate=0.2):
    inp = Input(shape=(input_dim,))

    ##  Encoder
    h = Dense(50, activation='relu')(inp)
    h = BatchNormalization()(h)
    h = Dropout(dropout_rate)(h)

    h = Dense(50, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dropout(dropout_rate)(h)

    h = Dense(200, activation='relu')(h)
    h = BatchNormalization()(h)
    h = Dropout(dropout_rate)(h)

    ## Latent space
    latent = Dense(latent_dim, activation='relu', name='latent')(h)

    ## Decoder
    h_dec = Dense(200, activation='relu')(latent)
    h_dec = BatchNormalization()(h_dec)
    h_dec = Dropout(dropout_rate)(h_dec)

    h_dec = Dense(50, activation='relu')(h_dec)
    h_dec = BatchNormalization()(h_dec)
    h_dec = Dropout(dropout_rate)(h_dec)

    h_dec = Dense(50, activation='relu')(h_dec)
    h_dec = BatchNormalization()(h_dec)
    h_dec = Dropout(dropout_rate)(h_dec)

    decoded = Dense(input_dim, activation='linear')(h_dec)

    ## Models
    ae = Model(inp, decoded, name='iDEC_autoencoder')
    encoder = Model(inp, latent, name='iDEC_encoder')

    ae.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
    return ae, encoder

## Trainig iDEC
def train_idec(X, ids, n_clusters, tag):
    print(f"\n--- Training iDEC for K={n_clusters} ({tag}) ---")
    input_dim = X.shape[1]
    ae, encoder = build_autoencoder(input_dim, LATENT_DIM)
    ## Initial latent
    latent = encoder.predict(X)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=SEED).fit(latent)
    cluster_centers = kmeans.cluster_centers_.astype(np.float32)
    optimizer = Adam(LEARNING_RATE)

    loss_history = []

    for iteration in range(N_ITER):
        q = soft_assignment(latent, cluster_centers, ALPHA)
        p = target_distribution(q)
        p_tf = tf.convert_to_tensor(p, dtype=tf.float32)

        with tf.GradientTape() as tape:
            ## Forward pass
            z = encoder(X, training=True)
            decoded = ae(X, training=True)
            
            ## Recontruction loss
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(decoded - X), axis=1))
            
            ## Softassignment computation
            centers_tf = tf.convert_to_tensor(cluster_centers, dtype=tf.float32)
            dist = tf.reduce_sum(tf.square(tf.expand_dims(z, 1) - centers_tf), axis=2)
            q_tf = 1.0 / (1.0 + dist / ALPHA)
            q_tf = q_tf ** ((ALPHA + 1.0) / 2.0)
            q_tf = q_tf / tf.reduce_sum(q_tf, axis=1, keepdims=True)
            
            ## Clustering loss
            clustering_loss = tf.reduce_mean(tf.reduce_sum(p_tf * tf.math.log(p_tf / (q_tf + 1e-10)), axis=1))
            
            ## Total loss
            loss = recon_loss + GAMMA * clustering_loss

        grads = tape.gradient(loss, ae.trainable_weights)
        optimizer.apply_gradients(zip(grads, ae.trainable_weights))

        latent = encoder.predict(X)
        try:
            loss_val = float(loss.numpy())
        except:
            loss_val = np.nan
        loss_history.append(loss_val)

        if (iteration + 1) % UPDATE_INTERVAL == 0:
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, init=cluster_centers, random_state=SEED).fit(latent)
            cluster_centers = kmeans.cluster_centers_.astype(np.float32)
            print(f"Iteration {iteration+1}: loss={loss_history[-1]:.4f}")

    ## Final soft assingments
    q_final = soft_assignment(latent, cluster_centers, ALPHA)
    y_pred = q_final.argmax(axis=1)
    sil = silhouette_score(latent, y_pred) if len(np.unique(y_pred)) > 1 else np.nan

    ## Save latent CSV
    latent_df = pd.DataFrame(latent, columns=[f"latent{i+1}" for i in range(latent.shape[1])])
    latent_df.insert(0, 'ID', ids)
    latent_df['cluster'] = y_pred
    latent_csv = os.path.join(OUTPUTS_DIR, f"latent_{tag}_with_clusters.csv")
    latent_df.to_csv(latent_csv, index=False)
    print(f"Latent embeddings written to {latent_csv}")

    ## Write soft assignments
    q_df = pd.DataFrame(q_final, columns=[f"cluster_{k}" for k in range(q_final.shape[1])])
    q_df.insert(0, "ID", ids)
    q_csv = os.path.join(OUTPUTS_DIR, f"soft_assignments_{tag}.csv")
    q_df.to_csv(q_csv, index=False)
    print(f"Soft assignments written to {q_csv}")

    ## Write loss history
    loss_df = pd.DataFrame({"iteration": list(range(1, len(loss_history) + 1)), "loss": loss_history})
    loss_csv = os.path.join(OUTPUTS_DIR, f"loss_history_{tag}.csv")
    loss_df.to_csv(loss_csv, index=False)
    print(f"Loss history written to {loss_csv}")

    ## Write cluster centers
    centers_csv = os.path.join(OUTPUTS_DIR, f"cluster_centers_{tag}.csv")
    pd.DataFrame(cluster_centers).to_csv(centers_csv, index=False)
    print(f"Cluster centers written to {centers_csv}")

    ## Plots
    plot_embedding(latent, y_pred, ids, tag)
    plot_soft_cluster_assignments(q_final, y_pred, ids, tag)

    metrics = {'final_loss': loss_history[-1] if loss_history else np.nan, 'silhouette': sil, 'chosen_k': n_clusters}
    return latent, y_pred, q_final, metrics

## Load data
df = pd.read_csv(CSV_PATH)
ids = df['ID'].astype(str).values
col_sets = {}
if any(c.startswith('morph_') for c in df.columns):
    col_sets['morph'] = [c for c in df.columns if c.startswith('morph_')]
if any(c.startswith('dna_') for c in df.columns):
    col_sets['dna'] = [c for c in df.columns if c.startswith('dna_')]
if any(c.startswith('env_') for c in df.columns):
    col_sets['env'] = [c for c in df.columns if c.startswith('env_')]

## Combined matrix
all_cols = [c for cols in col_sets.values() for c in cols]
if not all_cols:
    raise RuntimeError("No modality columns found (morph_*, dna_*, env_*).")
X_orig = df[all_cols].values.astype(np.float32)
X_orig = StandardScaler().fit_transform(X_orig)

## Ablation sets
modal_names = list(col_sets.keys())
ablation_sets = []
ablation_sets.append(tuple(modal_names)) 
for comb in itertools.combinations(modal_names, 2):
    ablation_sets.append(tuple(comb))
for m in modal_names:
    ablation_sets.append((m,))

print("Ablation runs (in order):", ablation_sets)

## K selection (full model)
sil_scores = {}
for k in K_RANGE:
    latent, y_pred, q_final, metrics = train_idec(X_orig, ids, n_clusters=k, tag=f"k_selection_full_K{k}")
    sil_scores[k] = metrics['silhouette']
    print(f"K={k}: silhouette={metrics['silhouette']:.4f}")

## Write silhouette scores CSV
sil_df = pd.DataFrame({"K": list(K_RANGE), "silhouette": [sil_scores[k] for k in K_RANGE]})
sil_csv_path = os.path.join(OUTPUTS_DIR, f"silhouette_scores_full_model.csv")
sil_df.to_csv(sil_csv_path, index=False)
print(f"Silhouette scores written to {sil_csv_path}")

## Determine optimal K
optimal_k = max(sil_scores, key=sil_scores.get)
print(f"\nOptimal K={optimal_k} (Silhouette={sil_scores[optimal_k]:.4f})")

## Plot K-selection
k_selection_plot = os.path.join(PLOTS_DIR, "k_selection_full_model.pdf")
plot_k_scores = {k: sil_scores[k] for k in sil_scores}

fig, ax = plt.subplots(figsize=(7, 5))
Klist = list(K_RANGE)
sil_vals = [plot_k_scores[k] for k in Klist]
ax.plot(Klist, sil_vals, marker='o', linestyle='-')
ax.scatter(optimal_k, plot_k_scores[optimal_k], color='r', s=80)
ax.set_xticks(Klist)
ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Silhouette score")
ax.set_title("K-selection (silhouette)")
ax.grid(False)

plt.tight_layout()
plt.savefig(k_selection_plot, bbox_inches='tight')
plt.close()
print(f"Written K-selection plot to {k_selection_plot}")

## Final full run with optimal K
latent_full, y_pred_full, q_final_full, metrics_full = train_idec(X_orig, ids, n_clusters=optimal_k, tag="full_FINAL")
print("Full model metrics:", metrics_full)

## Ablation runs
summary_rows = []
summary_rows.append({'ablation': 'full', 'ari_vs_full': 1.0, 'nmi_vs_full': 1.0, 'silhouette': metrics_full['silhouette'], 'chosen_k': optimal_k})

for ab_set in ablation_sets[1:]:
    col_to_idx = {c: i for i, c in enumerate(all_cols)}
    mask = np.ones_like(X_orig)
    for m in modal_names:
        if m not in ab_set:
            idxs = [col_to_idx[c] for c in col_sets[m]]
            mask[:, idxs] = 0
    X_ab = X_orig * mask
    tag = "_".join(ab_set)
    latent_ab, y_pred_ab, q_final_ab, metrics_ab = train_idec(X_ab, ids, n_clusters=optimal_k, tag=tag)
    ari = adjusted_rand_score(y_pred_full, y_pred_ab)
    nmi = normalized_mutual_info_score(y_pred_full, y_pred_ab)
    print(f"Ablation {tag}: ARI={ari:.4f}, NMI={nmi:.4f}, silhouette={metrics_ab['silhouette']:.4f}")
    summary_rows.append({'ablation': tag, 'ari_vs_full': ari, 'nmi_vs_full': nmi, 'silhouette': metrics_ab['silhouette'], 'chosen_k': optimal_k})

## Write ablation summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUTS_DIR, "ablation_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Ablation summary written to {summary_csv}")

# Write ARI and NMI bar charts
barchart_path = os.path.join(PLOTS_DIR, "ablation_ari_nmi_barchart.pdf")
plot_ari_nmi_barchart(summary_df, barchart_path)
summary_Write_path = os.path.join(OUTPUTS_DIR, "ablation_summary_for_plot.csv")
summary_df.to_csv(summary_Write_path, index=False)
print(f"Written ARI and NMI bar charts to {barchart_path} and summary CSV to {summary_Write_path}")

print("All outputs written to", OUTPUTS_DIR, "and plots to", PLOTS_DIR)
