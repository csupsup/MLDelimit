#!/usr/bin/env python3

import os
import argparse
import itertools
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
from matplotlib.cm import get_cmap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam
import random

####################
## iDEC Multimodal
####################

## Command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="iDEC multimodal clustering for joint species delimitation.")
    parser.add_argument("--input", required=True, help="Input CSV containing morphology, environment, and dna.")
    parser.add_argument("--plots_dir", default="plots", help="Directory to write plots.")
    parser.add_argument("--outputs_dir", default="outputs", help="Directory to write latent space, cluster assingments, and other outputs.")
    
    ## Hyperparameters
    parser.add_argument("--latent_per_mod", type=int, default=8, help="Latent dimension per modality.")
    parser.add_argument("--n_iter", type=int, default=50, help="Number of iDEC training iterations.")
    parser.add_argument("--update_interval", type=int, default=10, help="Update interval for cluster centers.")
    parser.add_argument("--gamma", type=float, default=0.1, help="GAMMA clustering loss weight.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha parameter for soft assignment.")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")

    ## K-selection to initialize iDEC
    parser.add_argument("--k_min", type=int, default=2, help="Minimum K to test.")
    parser.add_argument("--k_max", type=int, default=5, help="Maximum K to test.")

    ## Plot legend wrapping
    parser.add_argument("--wrap_width", type=int, default=30, help="Line-wrap width for legend labels.")

    ## Random seeds
    parser.add_argument("--seed", type=int, default=1225, help="Random seed for reproducibility.")

    return parser.parse_args()

args = parse_args()

## Apply arguments (xtian feels really lazy today to edit this redundant section :) haha)
PLOTS_DIR = args.plots_dir
OUTPUTS_DIR = args.outputs_dir
CSV_PATH = args.input

LATENT_PER_MOD = args.latent_per_mod
N_ITER = args.n_iter
UPDATE_INTERVAL = args.update_interval
GAMMA = args.gamma
ALPHA = args.alpha
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size

K_RANGE = range(args.k_min, args.k_max)
WRAP_WIDTH = args.wrap_width

SEED = args.seed
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("\n--- iDEC Mutimodal Settings ---")
print(f"Input CSV:        {CSV_PATH}")
print(f"Plots Dir:        {PLOTS_DIR}")
print(f"Outputs Dir:      {OUTPUTS_DIR}")
print(f"Latent/Mod:       {LATENT_PER_MOD}")
print(f"Iterations:       {N_ITER}")
print(f"Update Interval:  {UPDATE_INTERVAL}")
print(f"Gamma:            {GAMMA}")
print(f"Alpha:            {ALPHA}")
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
    return (q / np.sum(q, axis=1, keepdims=True)).astype(np.float32)

def target_distribution(q):
    weight = q ** 2 / q.sum(axis=0)
    return (weight.T / weight.sum(axis=1)).T.astype(np.float32)

## Build autoencoder
def build_modality_autoencoder(modalities_dict, latent_per_mod=10):
    """
    Autoencoder model with modalities
    """
    concat_order = list(modalities_dict.keys())
    inputs = {}
    encodings = {}
    decodings = {}
    latent_dims = {}
    for mod_name in concat_order:
        X_mod = modalities_dict[mod_name]['X']
        in_dim = X_mod.shape[1]
        inp = Input(shape=(in_dim,), name=f"{mod_name}_input")
        inputs[mod_name] = inp
        ## Encoder
        if mod_name == 'dna':
            h = Dense(50, activation='relu')(inp)
        else:
            h = Dense(20, activation='relu')(inp)
        h = BatchNormalization()(h)
        h = Dense(latent_per_mod, activation='relu')(h)
        encodings[mod_name] = h
        latent_dims[mod_name] = latent_per_mod

    ## Concatenate
    if len(encodings) == 1:
        sole = list(encodings.keys())[0]
        latent = encodings[sole]
    else:
        latent = Concatenate(name='latent_concat')([encodings[m] for m in concat_order])
    
    ## Decoders
    for mod_name in concat_order:
        out_dim = modalities_dict[mod_name]['X'].shape[1]
        if mod_name == 'dna':
            d = Dense(50, activation='relu')(latent)
        else:
            d = Dense(20, activation='relu')(latent)
        d = Dense(out_dim, activation='linear', name=f"decoded_{mod_name}")(d)
        decodings[mod_name] = d

    ae_inputs = [inputs[m] for m in concat_order]
    ae_outputs = [decodings[m] for m in concat_order]
    autoencoder = Model(inputs=ae_inputs, outputs=ae_outputs)
    encoder = Model(inputs=ae_inputs, outputs=latent)
    autoencoder.compile(optimizer=Adam(LEARNING_RATE), loss='mse')
    return autoencoder, encoder, ae_inputs, concat_order, latent_dims

def plot_embedding_combined(latent, ids, y_pred, y_true, tag):
    """
    Write PCA and t-SNE plots with ellipses and other outputs.
    """
    unique_clusters = np.unique(y_pred)
    n_clusters = len(unique_clusters)

    ## Set colors
    if n_clusters <= 10:
        cmap = get_cmap('tab10')  
    elif n_clusters <= 20:
        cmap = get_cmap('tab20')  
    else:
        cmap = get_cmap('hsv')

    cluster_colors = [cmap(i / max(n_clusters-1, 1)) for i in range(n_clusters)]
    cluster_color_map = {cid: cluster_colors[i] for i, cid in enumerate(unique_clusters)}

    le_local = LabelEncoder()
    y_true_enc = le_local.fit_transform(y_true) if y_true is not None else None
    unique_true = np.unique(y_true_enc) if y_true is not None else []

    ## Colors with true IDs
    n_true = len(unique_true)
    if n_true <= 10:
        id_cmap = get_cmap('tab10')
    elif n_true <= 20:
        id_cmap = get_cmap('tab20')
    else:
        id_cmap = get_cmap('hsv')
    id_colors = [id_cmap(i / max(n_true-1, 1)) for i in range(n_true)]
    id_color_map = {uid: id_colors[i] for i, uid in enumerate(unique_true)}

    def _plot_and_write(embedding, name_suffix):
        fig, axes = plt.subplots(1, 2, figsize=(18,8))
        ## Left: clusters
        ax = axes[0]
        for i, (x,y) in enumerate(embedding):
            c = y_pred[i]
            ax.scatter(x, y, color=cluster_color_map[c], s=50, alpha=0.7)
        ## Ellipses per cluster
        for cluster_id in unique_clusters:
            points = embedding[y_pred == cluster_id]
            if len(points) > 1:
                mean = points.mean(axis=0)
                cov = np.cov(points.T)
                vals, vecs = np.linalg.eigh(cov)
                width, height = 2 * np.sqrt(np.clip(vals, 1e-8, None)) 
                angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                                  alpha=0.2, color=cluster_color_map[cluster_id])
                ax.add_patch(ellipse)
        ## Wrapped legend
        legend_elements_clusters = [
            Patch(color=cluster_color_map[cid],
                  label="\n".join(textwrap.wrap(
                      f"Cluster {cid}: " + ", ".join(ids[y_pred==cid]),
                      width=WRAP_WIDTH)))
            for cid in unique_clusters
        ]
        ax.legend(handles=legend_elements_clusters, bbox_to_anchor=(1.05,1), loc='upper left',
                  fontsize=8, frameon=False)
        ax.set_title(f"iDEC Clusters ({tag})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(False)

        ## Right: true IDs
        ax = axes[1]
        for i, (x,y) in enumerate(embedding):
            uid = y_true_enc[i]
            ax.scatter(x, y, color=id_color_map[uid], s=50, alpha=0.7)
        ## ellipses per true id
        for uid in unique_true:
            points = embedding[y_true_enc == uid]
            if len(points) > 1:
                mean = points.mean(axis=0)
                cov = np.cov(points.T)
                vals, vecs = np.linalg.eigh(cov)
                width, height = 2 * np.sqrt(np.clip(vals, 1e-8, None))
                angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                                  alpha=0.2, color=id_color_map[uid])
                ax.add_patch(ellipse)
        ## Wrapped legend
        legend_elements_true = [
            Patch(color=id_color_map[uid],
                  label="\n".join(textwrap.wrap(
                      f"ID {le_local.inverse_transform([uid])[0]}: " + ", ".join(ids[y_true_enc==uid]),
                      width=WRAP_WIDTH)))
            for uid in unique_true
        ]
        ax.legend(handles=legend_elements_true, bbox_to_anchor=(1.05,1), loc='upper left',
                  fontsize=8, frameon=False)
        ax.set_title(f"True IDs ({tag})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(False)

        plt.tight_layout()
        outname = os.path.join(PLOTS_DIR, f"{name_suffix}_{tag}.pdf")
        plt.savefig(outname, bbox_inches='tight')
        plt.close()

    ## PCA
    pca_emb = PCA(n_components=2).fit_transform(latent)
    
    ## Write PCA embeddings to CSV
    pca_df = pd.DataFrame(pca_emb, columns=["PC1","PC2"])
    pca_df.insert(0, "ID", ids)
    pca_df["cluster"] = y_pred
    if y_true is not None:
        pca_df["true_id"] = y_true
    pca_csv_path = os.path.join(OUTPUTS_DIR, f"pca_embedding_{tag}.csv")
    pca_df.to_csv(pca_csv_path, index=False)
    print(f"PCA embeddings written to {pca_csv_path}")
    _plot_and_write(pca_emb, "latent_pca_combined_ellipses")

    ## t-SNE
    tsne_emb = TSNE(n_components=2, random_state=SEED).fit_transform(latent)
    tsne_df = pd.DataFrame(tsne_emb, columns=["TSNE1","TSNE2"])
    tsne_df.insert(0, "ID", ids)
    tsne_df["cluster"] = y_pred
    if y_true is not None:
        tsne_df["true_id"] = y_true
    tsne_csv_path = os.path.join(OUTPUTS_DIR, f"tsne_embedding_{tag}.csv")
    tsne_df.to_csv(tsne_csv_path, index=False)
    print(f"t-SNE embeddings written to {tsne_csv_path}")
    _plot_and_write(tsne_emb, "latent_tsne_combined_ellipses")

## Soft cluster assignments
def plot_soft_cluster_assignments_all_samples_cb(q, y_pred, ids, filename):
    """
    Soft cluster assignments
    """
    n_samples, n_clusters = q.shape

    ## Write q matrix to CSV
    q_df = pd.DataFrame(q, columns=[f"cluster_{k}" for k in range(n_clusters)])
    q_df.insert(0, "ID", ids)
    q_csv = os.path.join(OUTPUTS_DIR, os.path.basename(filename).replace('.pdf', '_soft_assignments.csv'))
    q_df.to_csv(q_csv, index=False)
    print(f"Soft assignment matrix written to {q_csv}")

    ## Sort samples
    sort_idx = np.argsort(y_pred)
    q_sorted = q[sort_idx]
    ids_sorted = ids[sort_idx]

    ## Set colors
    cmap = plt.get_cmap("tab20")
    cluster_colors = [cmap(i % 20) for i in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(max(12, n_samples/5), 6)) 

    ## Plot
    for i in range(n_samples):
        probs = q_sorted[i]
        sorted_idx = np.argsort(-probs)
        bottom = 0
        for k in sorted_idx:
            ax.bar(i, probs[k], bottom=bottom, color=cluster_colors[k], width=0.8)
            bottom += probs[k]

    ax.set_xlabel("Samples (sorted by cluster)")
    ax.set_ylabel("Soft Assignment Probability")
    ax.set_title(f"Soft Cluster Assignments (K={n_clusters})")

    ## Show all IDs
    ax.set_xticks(range(n_samples))
    ax.set_xticklabels(ids_sorted, rotation=90, fontsize=6)

    ## Legend for clusters
    legend_handles = [Patch(color=cluster_colors[k], label=f'Cluster {k}') for k in range(n_clusters)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(False) 

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

## K-Selection plot
def plot_k_selection(silhouette_scores, K_RANGE, outname):
    """
    Plots the silhouette scores vs K to visualize the optimal K selection
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    K = list(K_RANGE)
    sil_vals = [silhouette_scores[k] for k in K]
    
    ## Find optimal K
    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    max_sil = silhouette_scores[optimal_k]
    
    ax.plot(K, sil_vals, marker='o', linestyle='-', color='b')
    ax.scatter(optimal_k, max_sil, color='r', s=100, label=f'Optimal K={optimal_k}')
    ax.text(optimal_k, max_sil, f"Max: {max_sil:.3f}", ha='left', va='bottom', color='r')
    
    ax.set_xticks(K)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score (Maximize)")
    ax.set_title("Optimal K Selection using Silhouette Score")
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(outname, bbox_inches='tight')
    plt.close()

    ## Write silhouette scores to CSV
    sil_df = pd.DataFrame({
        "K": K,
        "silhouette": [silhouette_scores[k] for k in K]
    })
    sil_csv = os.path.join(OUTPUTS_DIR, os.path.basename(outname).replace('.pdf', '_silhouette_scores.csv'))
    sil_df.to_csv(sil_csv, index=False)
    print(f"Silhouette scores written to {sil_csv}")

def train_run_idec(modalities_dict, tag, ids, n_clusters, y_true=None, 
                   n_iter=N_ITER, update_interval=UPDATE_INTERVAL, plot_results=True):
    """
    iDEC training with fixed GAMMA (clustering loss weight)
    """
    print(f"\n--- Training iDEC for: {tag} with K={n_clusters} ---")
    
    autoencoder, encoder, ae_inputs, concat_order, latent_dims = build_modality_autoencoder(
        modalities_dict, latent_per_mod=LATENT_PER_MOD
    )
    
    X_inputs = [modalities_dict[m]['X'] for m in concat_order]

    ## Initial latent
    latent_feats = encoder.predict(X_inputs)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=SEED)
    kmeans.fit(latent_feats)
    cluster_centers = kmeans.cluster_centers_.astype(np.float32)

    optimizer = Adam(LEARNING_RATE)
    latent = latent_feats.copy()
    cluster_history = []
    loss_history = []

    for iteration in range(n_iter):
        q = soft_assignment(latent, cluster_centers, ALPHA)
        p = target_distribution(q)
        p_tf = tf.convert_to_tensor(p, dtype=tf.float32)

        with tf.GradientTape() as tape:
            decoded_preds = autoencoder(X_inputs, training=True)
            if len(concat_order) == 1:
                decoded_preds = [decoded_preds]
            decoded_dict = {m: pred for m, pred in zip(concat_order, decoded_preds)}
            z = encoder(X_inputs, training=True)

            centers_tf = tf.convert_to_tensor(cluster_centers, dtype=tf.float32)
            expanded = tf.expand_dims(z, 1)
            dist = tf.reduce_sum(tf.square(expanded - centers_tf), axis=2)
            q_tf = 1.0 / (1.0 + dist/ALPHA)
            q_tf = q_tf ** ((ALPHA+1.0)/2.0)
            q_tf = q_tf / tf.reduce_sum(q_tf, axis=1, keepdims=True)

            ## Clustering loss (KL divergence)
            clustering_loss = tf.reduce_mean(tf.reduce_sum(p_tf * tf.math.log(p_tf / (q_tf + 1e-10)), axis=1))

            ## Reconstruction loss
            recon_loss_total = 0.0
            for m in concat_order:
                X_m = tf.convert_to_tensor(modalities_dict[m]['X'], dtype=tf.float32)
                recon_m = tf.reduce_mean(tf.reduce_sum(tf.square(decoded_dict[m] - X_m), axis=1))
                recon_loss_total += recon_m

            ## Total loss with fixed GAMMA
            loss = recon_loss_total + GAMMA * clustering_loss

        grads = tape.gradient(loss, autoencoder.trainable_weights)
        optimizer.apply_gradients(zip(grads, autoencoder.trainable_weights))

        latent = encoder.predict(X_inputs)
        try:
            loss_val = float(loss.numpy())
        except:
            loss_val = np.nan
        loss_history.append(loss_val)

        if (iteration + 1) % update_interval == 0:
            kmeans = KMeans(n_clusters=n_clusters, n_init=1, init=cluster_centers, random_state=SEED)
            kmeans.fit(latent)
            cluster_centers = kmeans.cluster_centers_.astype(np.float32)
            y_temp = soft_assignment(latent, cluster_centers, ALPHA).argmax(axis=1)
            cluster_history.append(y_temp.copy())
            print(f"Iteration {iteration+1}: loss={loss_val:.4f}")

    ## Final assignment
    q_final = soft_assignment(latent, cluster_centers, ALPHA)
    y_pred = q_final.argmax(axis=1)
    sil = silhouette_score(latent, y_pred) if len(np.unique(y_pred)) > 1 else np.nan

    ## Write latent CSV and plots
    if plot_results:
        latent_cols = [f"latent{i+1}" for i in range(latent.shape[1])]
        latent_df = pd.DataFrame(latent, columns=latent_cols)
        latent_df.insert(0, "ID", ids)
        latent_df["cluster"] = y_pred
        latent_csv = os.path.join(OUTPUTS_DIR, f"latent_{tag}_with_clusters.csv")
        latent_df.to_csv(latent_csv, index=False)
        print(f"Latent embeddings written to {latent_csv}")

        ## Write soft assignments and loss history and cluster history
        q_df = pd.DataFrame(q_final, columns=[f"cluster_{k}" for k in range(q_final.shape[1])])
        q_df.insert(0, "ID", ids)
        q_csv = os.path.join(OUTPUTS_DIR, f"soft_assignments_{tag}.csv")
        q_df.to_csv(q_csv, index=False)
        print(f"Soft assignments written to {q_csv}")

        loss_df = pd.DataFrame({"iteration": list(range(1, len(loss_history)+1)), "loss": loss_history})
        loss_csv = os.path.join(OUTPUTS_DIR, f"loss_history_{tag}.csv")
        loss_df.to_csv(loss_csv, index=False)
        print(f"Loss history written to {loss_csv}")

        ## Cluster history: write as CSV if non-empty
        if cluster_history:
            ch_df = pd.DataFrame(cluster_history).T
            ## Columns recorded at update steps (1..)
            ch_csv = os.path.join(OUTPUTS_DIR, f"cluster_history_{tag}.csv")
            ch_df.insert(0, "ID", ids)
            ch_df.to_csv(ch_csv, index=False)
            print(f"Cluster history written to {ch_csv}")

        if y_true is None:
            y_true = ids
        plot_embedding_combined(latent, ids, y_pred, y_true, tag)

    metrics = {"chosen_k": n_clusters, "silhouette": sil, "final_loss": loss_history[-1] if loss_history else np.nan}

    return latent, y_pred, q_final, metrics

## Load data and build full modalities
df = pd.read_csv(CSV_PATH)
ids = df['ID'].values.astype(str)

## Identify modalities
morph_cols = [c for c in df.columns if c.startswith("morph_")]
env_cols = [c for c in df.columns if c.startswith("env_")]
dna_cols = [c for c in df.columns if c.startswith("dna_")]

modalities_full = {}
if morph_cols:
    X_morph = df[morph_cols].values.astype(np.float32)
    X_morph = StandardScaler().fit_transform(X_morph)
    modalities_full['morph'] = {'X': X_morph, 'cols': morph_cols}
if env_cols:
    X_env = df[env_cols].values.astype(np.float32)
    X_env = StandardScaler().fit_transform(X_env)
    modalities_full['env'] = {'X': X_env, 'cols': env_cols}
if dna_cols:
    X_dna = df[dna_cols].values.astype(np.float32)
    X_dna = StandardScaler().fit_transform(X_dna)
    modalities_full['dna'] = {'X': X_dna, 'cols': dna_cols}

if len(modalities_full) == 0:
    raise RuntimeError("No modality columns found (morph_*, env_*, dna_*) in CSV.")

## Define all ablation sets (full, pairwise, singles)
modal_names = list(modalities_full.keys())  ## e.g. ['morph','env','dna']
desired_order = []
desired_order.append(tuple(modal_names)) 
## Pairwise
for comb in itertools.combinations(modal_names, 2):
    desired_order.append(tuple(comb))
## Singles
for m in modal_names:
    desired_order.append((m,))
ablation_sets = desired_order

print("Ablation runs (in order):", ablation_sets)

## Find Optimal K for Full Model
full_tag = "_".join(sorted(ablation_sets[0]))
print(f"\n--- K-Selection: Finding Optimal K for Full iDEC Model ({full_tag}) ---")

silhouette_scores = {}
for k in K_RANGE:
    ## Run iDEC for each K, but skip plotting to write time
    _, _, _, metrics_k = train_run_idec(
        {m: modalities_full[m] for m in ablation_sets[0]}, 
        tag=f"k_selection_{full_tag}", 
        ids=ids, 
        n_clusters=k, 
        y_true=ids,
        plot_results=False
    )
    
    silhouette_scores[k] = metrics_k['silhouette']
    print(f"K={k}: Silhouette Score = {metrics_k['silhouette']:.4f}")

## Write silhouette scores CSV
sil_csv_all = os.path.join(OUTPUTS_DIR, f"silhouette_scores_{full_tag}.csv")
pd.DataFrame({
    "K": list(K_RANGE),
    "silhouette": [silhouette_scores[k] for k in K_RANGE]
}).to_csv(sil_csv_all, index=False)
print(f"Written silhouette scores summary to {sil_csv_all}")

## Determine optimal K
optimal_k_full = max(silhouette_scores, key=silhouette_scores.get)
best_k_sil = silhouette_scores[optimal_k_full]
print(f"\nOptimal K for Full Model selected: {optimal_k_full} (Silhouette: {best_k_sil:.4f})")

## Plot K-selection results and write to CSV
plot_k_selection(silhouette_scores, K_RANGE, os.path.join(PLOTS_DIR, f"k_selection_full_model.pdf"))

## FINAL RUN: Full Model with Optimal K
print(f"\n--- FINAL RUN: Full Model with Optimal K={optimal_k_full} ---")
## Run the full model one last time with the optimal K and enable plotting
latent_full, y_pred_full, q_final_full, metrics_full = train_run_idec(
    {m: modalities_full[m] for m in ablation_sets[0]}, 
    tag=f"full_FINAL_{full_tag}", 
    ids=ids, 
    n_clusters=optimal_k_full, 
    y_true=ids,
    plot_results=True ## Enable plotting for the final result
)
print("Full model final metrics:", metrics_full)

## Plot Stacked Soft Assignment Bar Plot
soft_plot_filename = os.path.join(PLOTS_DIR, f"soft_cluster_assignments_cb_full_FINAL_{full_tag}.pdf")
plot_soft_cluster_assignments_all_samples_cb(q_final_full, y_pred_full, ids, soft_plot_filename)
print(f"Color-blind friendly stacked bar plot written to '{soft_plot_filename}'")

## RUN Ablations and Compare to Full Model
summary_rows = []

## Tag for the full model
full_ablation_tag = f"full_{full_tag}"

## Include full model first
summary_rows.append({
    "ablation": full_ablation_tag,
    "ari_vs_full": 1.0,
    "nmi_vs_full": 1.0,
    "silhouette": metrics_full['silhouette'],
    "chosen_k": metrics_full['chosen_k']
})

for ab_set in ablation_sets[1:]:
    ## Skip any ablation that exactly matches the full modalities
    if set(ab_set) == set(ablation_sets[0]):
        print(f"Skipping ablation {ab_set} because it matches the full modalities.")
        continue

    ## Tag for CSV and plotting
    ab_tag = "_".join(sorted(ab_set))

    ## Prepare input for ablation
    modal_dict = {k: modalities_full[k] for k in ab_set}

    ## Train iDEC on ablation
    latent_ab, y_pred_ab, q_final_ab, metrics_ab = train_run_idec(
        modal_dict,
        tag=ab_tag,
        ids=ids,
        n_clusters=optimal_k_full,
        y_true=ids,
        plot_results=True
    )

    ## Compute ARI and NMI vs full
    ari_vs_full = adjusted_rand_score(y_pred_full, y_pred_ab)
    nmi_vs_full = normalized_mutual_info_score(y_pred_full, y_pred_ab)

    sil = metrics_ab['silhouette']
    chosen_k = metrics_ab['chosen_k']

    print(f"Ablation {ab_tag}: ARI vs full = {ari_vs_full:.4f}, NMI vs full = {nmi_vs_full:.4f}, silhouette = {sil:.4f}, k={chosen_k}")

    summary_rows.append({
        "ablation": ab_tag,
        "ari_vs_full": ari_vs_full,
        "nmi_vs_full": nmi_vs_full,
        "silhouette": sil,
        "chosen_k": chosen_k
    })

## Write summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUTS_DIR, "ablation_summary_ari_nmi_silhouette.csv")
summary_df.to_csv(summary_csv, index=False)
print("Ablation summary written to", summary_csv)

## ARI/NMI Bar Charts
def write_ari_nmi_barchart(summary_df, outpath):
    """
    Plot ARI and NMI bar charts for ablations vs full model.
    """
    ## Only keep ablations that differ from full
    plot_df = summary_df[summary_df['ari_vs_full'] < 1.0].reset_index(drop=True)
    
    if plot_df.empty:
        print("No ablations to plot (all identical to full). Skipping ARI/NMI bar chart.")
        return

    labels = plot_df['ablation'].tolist()
    ari_vals = plot_df['ari_vs_full'].tolist()
    nmi_vals = plot_df['nmi_vs_full'].tolist()
    x = np.arange(len(labels))
    width = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    ## ARI
    ax = axes[0]
    bars = ax.bar(x, ari_vals, width, edgecolor='black')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title("ARI vs Full Model")
    ax.set_ylabel("Adjusted Rand Index")
    ax.grid(False)
    for i, v in enumerate(ari_vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    ## NMI
    ax = axes[1]
    bars = ax.bar(x, nmi_vals, width, edgecolor='black')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title("NMI vs Full Model")
    ax.set_ylabel("Normalized Mutual Information")
    ax.grid(False)
    for i, v in enumerate(nmi_vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"Written ARI and NMI bar chart to {outpath}")

## Write bar chart
barchart_path = os.path.join(PLOTS_DIR, "ablation_ari_nmi_barchart.pdf")
write_ari_nmi_barchart(summary_df, barchart_path)

print("All outputs written to", OUTPUTS_DIR, "and plots to", PLOTS_DIR)
