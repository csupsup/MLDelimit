#!/usr/bin/env python3

import os
import math
import textwrap
import itertools
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch

##################
## VaDE Delimit
##################

## Command line arguments
parser = argparse.ArgumentParser(description="VaDe clustering for joint species delimitation.")
parser.add_argument("--input", type=str, required=True, help="Input CSV file containing morphology, environment, and dna data.")
parser.add_argument("--id_col", type=str, default="ID", help="Column name for sample IDs.")
parser.add_argument("--latent_dim", type=int, default=7, help="Latent dimension size.")
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[50, 50, 200], help="Hidden layers sizes.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--pretrain_epochs", type=int, default=30, help="Pretraining epochs.")
parser.add_argument("--vade_epochs", type=int, default=100, help="VaDE training epochs.")
parser.add_argument("--pretrain_lr", type=float, default=1e-3, help="Learning rate for pretraining.")
parser.add_argument("--vade_lr", type=float, default=1e-3, help="Learning rate for VaDE.")
parser.add_argument("--beta_anneal_epochs", type=int, default=30, help="Beta annealing epochs.")
parser.add_argument("--plots_dir", type=str, default="./plots", help="Directory to write plots and CSVs outputs")
parser.add_argument("--is_binary", action="store_true", help="Whether input data is binary.")
parser.add_argument("--k_min", type=int, default=2, help="Minimum K to test in GMM BIC search.")
parser.add_argument("--k_max", type=int, default=10, help="Maximum K to test in GMM BIC search.")
parser.add_argument("--seed", type=int, default=1225, help="Set seed.")
args = parser.parse_args()

## Settings
os.makedirs(args.plots_dir, exist_ok=True)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("\nUsing device:", device)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
wrap_width = 40

## GMM parameters (default)
reg_covar_init = 1e-3
min_covar = 1e-3

print("\n--- VaDE Settings ---")
print(f"Input CSV:            {args.input}")
print(f"Plots Dir:            {args.plots_dir}")
print(f"ID Column:            {args.id_col}")
print(f"Latent Dim:           {args.latent_dim}")
print(f"Hidden Dims:          {args.hidden_dims}")
print(f"Batch Size:           {args.batch_size}")
print(f"Pretrain Epochs:      {args.pretrain_epochs}")
print(f"VaDE Epochs:          {args.vade_epochs}")
print(f"Pretrain LR:          {args.pretrain_lr}")
print(f"VaDE LR:              {args.vade_lr}")
print(f"Beta Anneal Epochs:   {args.beta_anneal_epochs}")
print(f"Binary Input:         {args.is_binary}")
print(f"GMM Reg Covar Init:   {reg_covar_init}")
print(f"GMM Min Covar:        {min_covar}")
print(f"Wrap Width:           {wrap_width}")
print(f"K Range:              {args.k_min}–{args.k_max-1}")
print(f"Seed:                 {args.seed}")
print(f"Device:               {device}")
print("---\n")

## Load data
print("Loading dataset...")
df = pd.read_csv(args.input)
ids = df[args.id_col].values
X = df.drop(columns=[args.id_col]).values.astype(np.float32)
n_samples, input_dim = X.shape
print(f"Loaded {n_samples} samples, input_dim={input_dim}")

if not args.is_binary:
    X = StandardScaler().fit_transform(X).astype(np.float32)

dataset = TensorDataset(torch.from_numpy(X))
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

## MLP encoder and decoder
def mlp_layers(in_dim, hidden_dims, out_dim=None, activation=nn.ReLU):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    if out_dim is not None:
        layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.shared = mlp_layers(input_dim, hidden_dims)
        last = hidden_dims[-1]
        self.fc_mu = nn.Linear(last, latent_dim)
        self.fc_logvar = nn.Linear(last, latent_dim)
    def forward(self, x):
        h = self.shared(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super().__init__()
        rev_hidden = list(hidden_dims[::-1])
        self.shared = mlp_layers(latent_dim, rev_hidden, out_dim=output_dim)
    def forward(self, z):
        return self.shared(z)

class PlainVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        xr = self.decoder(z)
        return xr, z, mu, logvar

## Pretrain VAE for K initialization
vae = PlainVAE(input_dim, args.hidden_dims, args.latent_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=args.pretrain_lr)

## Store VAE pretrain losses
pretrain_loss_history = []  

print("Pretraining VAE (reconstruction + KL only)...")
for epoch in range(1, args.pretrain_epochs + 1):
    vae.train()
    total_loss = 0
    for xb_tuple in loader:
        xb = xb_tuple[0].to(device)
        xr, z, mu, logvar = vae(xb)
        recon = F.binary_cross_entropy_with_logits(xr, xb, reduction='sum')/xb.size(0) if args.is_binary else F.mse_loss(xr, xb, reduction='sum')/xb.size(0)
        kl = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/xb.size(0)
        loss = recon + kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    pretrain_loss_history.append(total_loss)
    print(f"Pretrain Epoch {epoch}: Loss {total_loss:.6f}")

## Write pretrain loss history
pretrain_loss_df = pd.DataFrame({"epoch": np.arange(1,args.pretrain_epochs+1), "loss": pretrain_loss_history})
pretrain_loss_df.to_csv(os.path.join(args.plots_dir,"vae_pretrain_loss_history_full_FINAL.csv"), index=False)
print("VAE pretrain loss history saved.")

## Find optimal K using GMM BIC
vae.eval()
with torch.no_grad():
    mus = []
    for xb_tuple in loader:
        xb = xb_tuple[0].to(device)
        mu, _ = vae.encoder(xb)
        mus.append(mu.cpu().numpy())
    mus = np.vstack(mus)

best_bic = np.inf
best_k = args.k_min
for k in range(args.k_min, min(args.k_max + 1, mus.shape[0]+1)):
    gmm_test = GaussianMixture(n_components=k, reg_covar=reg_covar_init, random_state=1225)
    gmm_test.fit(mus)
    bic = gmm_test.bic(mus)
    if bic < best_bic:
        best_bic = bic
        best_k = k
print(f"Optimal K via BIC: {best_k}")

## Full VaDE model
class VaDE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_clusters):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        self.n_clusters = n_clusters
        self.pi_logit = nn.Parameter(torch.zeros(n_clusters))
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim)*0.01)
        self.logvar_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))
    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        xr = self.decoder(z)
        return xr, z, mu, logvar

model = VaDE(input_dim, args.hidden_dims, args.latent_dim, best_k).to(device)

## Pretrain VaDE
pretrain_optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=args.pretrain_lr)
vade_pretrain_loss_history = []

for epoch in range(1, args.pretrain_epochs + 1):
    model.train()
    total_loss = 0.0
    for xb_tuple in loader:
        xb = xb_tuple[0].to(device)
        xr, z, mu, logvar = model(xb)
        recon = F.binary_cross_entropy_with_logits(xr, xb, reduction='sum')/xb.size(0) if args.is_binary else F.mse_loss(xr, xb, reduction='sum')/xb.size(0)
        kl = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/xb.size(0)
        loss = recon + kl
        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()
        total_loss += loss.item()
    vade_pretrain_loss_history.append(total_loss)
    print(f"[Pretrain] Epoch {epoch}: Loss {total_loss:.6f}")

## Write pretrain loss
vade_pretrain_loss_df = pd.DataFrame({"epoch": np.arange(1,args.pretrain_epochs+1), "loss": vade_pretrain_loss_history})
vade_pretrain_loss_df.to_csv(os.path.join(args.plots_dir,"vade_pretrain_loss_history_full_FINAL.csv"), index=False)
print("VaDE pretrain loss history saved.")

## GMM Initialization using optimal K
model.eval()
with torch.no_grad():
    mus = []
    for xb_tuple in loader:
        xb = xb_tuple[0].to(device)
        mu, _ = model.encoder(xb)
        mus.append(mu.cpu().numpy())
    mus = np.vstack(mus)
    gmm = GaussianMixture(n_components=best_k, reg_covar=reg_covar_init, random_state=1225).fit(mus)
    cov_diag = np.clip(np.array([np.diag(cov) for cov in gmm.covariances_]), min_covar, None)
    model.pi_logit.copy_(torch.log(torch.tensor(gmm.weights_ + 1e-12, dtype=torch.float32)))
    model.mu_c.copy_(torch.tensor(gmm.means_, dtype=torch.float32))
    model.logvar_c.copy_(torch.log(torch.tensor(cov_diag, dtype=torch.float32)))
print("GMM initialization completed.")

## VaDE Training
optimizer = torch.optim.Adam(model.parameters(), lr=args.vade_lr)
vade_train_loss_history = []

for epoch in range(1, args.vade_epochs + 1):
    model.train()
    total_loss = 0.0
    beta = min(1.0, epoch / args.beta_anneal_epochs)
    for xb_tuple in loader:
        xb = xb_tuple[0].to(device)
        xr, z, mu, logvar = model(xb)
        batch = xb.size(0)
        recon_loss = F.binary_cross_entropy_with_logits(xr, xb, reduction="sum")/batch if args.is_binary else F.mse_loss(xr, xb, reduction="sum")/batch
        z_exp = z.unsqueeze(1)
        mu_c = model.mu_c.unsqueeze(0)
        logvar_c = model.logvar_c.unsqueeze(0)
        log_p_z_given_c = -0.5*(torch.sum(logvar_c + (z_exp - mu_c)**2 / torch.exp(logvar_c), dim=2) + args.latent_dim*math.log(2*math.pi))
        log_pi = torch.log_softmax(model.pi_logit, dim=0).unsqueeze(0)
        log_pzc = log_pi + log_p_z_given_c
        log_pz = torch.logsumexp(log_pzc, dim=1)
        const_term = args.latent_dim*(1+math.log(2*math.pi))
        E_log_q = -0.5*(const_term + torch.sum(logvar, dim=1))
        kl_per_sample = E_log_q - log_pz
        kl_term = torch.mean(kl_per_sample)
        loss_batch = recon_loss + beta*kl_term
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        total_loss += loss_batch.item()
    vade_train_loss_history.append(total_loss)
    if epoch % 10 == 0 or epoch == 1:
        print(f"VaDE Epoch {epoch}: Loss {total_loss:.6f} (β={beta:.2f})")

## Write full VaDE training loss
vade_train_loss_df = pd.DataFrame({"epoch": np.arange(1,args.vade_epochs+1), "loss": vade_train_loss_history})
vade_train_loss_df.to_csv(os.path.join(args.plots_dir,"vade_loss_history_full_FINAL.csv"), index=False)
print("VaDE training loss history saved.")

## Cluster extraction and plotting functions
def log_sum_exp(a, dim=-1):
    m,_ = a.max(dim=dim, keepdim=True)
    return m.squeeze(dim) + torch.log(torch.sum(torch.exp(a - m), dim=dim))

def extract_clusters(model, dataset, batch_size=args.batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latent_list = []
    with torch.no_grad():
        for xb_tuple in loader:
            xb = xb_tuple[0].to(device)
            mu, logvar = model.encoder(xb)
            latent_list.append(mu.cpu().numpy())
    latent = np.vstack(latent_list)
    Z_t = torch.from_numpy(latent).to(device).float()
    z_exp = Z_t.unsqueeze(1)
    mu_c = model.mu_c.unsqueeze(0)
    logvar_c = model.logvar_c.unsqueeze(0)
    log_p_z_given_c = -0.5*(torch.sum(logvar_c + (z_exp - mu_c)**2 / torch.exp(logvar_c), dim=2) + args.latent_dim*math.log(2*math.pi))
    log_pi = torch.log_softmax(model.pi_logit, dim=0).unsqueeze(0)
    log_pzc = log_pi + log_p_z_given_c
    log_q_c = log_pzc - log_sum_exp(log_pzc, dim=1).unsqueeze(1)
    q_c = torch.exp(log_q_c.cpu())
    clusters = torch.argmax(q_c, dim=1).numpy()
    return latent, clusters

def compute_clusters(model, loader):
    model.eval()
    zs = []
    with torch.no_grad():
        for xb_tuple in loader:
            xb = xb_tuple[0].to(device)
            _, z, _, _ = model(xb)
            zs.append(z.cpu().numpy())
    zs = np.vstack(zs)
    z_exp = zs[:,None,:]
    mu_c = model.mu_c.detach().cpu().numpy()[None,:,:]
    logvar_c = model.logvar_c.detach().cpu().numpy()[None,:,:]
    log_pi = F.log_softmax(model.pi_logit, dim=0).detach().cpu().numpy()[None,:]
    log_pzc = log_pi -0.5*np.sum(np.log(2*np.pi*np.exp(logvar_c)), axis=2) -0.5*np.sum((z_exp - mu_c)**2/np.exp(logvar_c), axis=2)
    clusters = np.argmax(log_pzc, axis=1)
    return zs, clusters

def plot_embedding_combined(latent, ids_array, y_pred, y_true=None, tag="full_model"):
    le_local = LabelEncoder()
    y_true_enc = le_local.fit_transform(y_true) if y_true is not None else None
    unique_clusters = np.unique(y_pred)
    unique_true = np.unique(y_true_enc) if y_true is not None else []

    def get_color_map(n):
        if n <= 10: return plt.get_cmap('tab10')
        elif n <= 20: return plt.get_cmap('tab20')
        else: return plt.get_cmap('hsv')

    cluster_cmap = get_color_map(len(unique_clusters))
    cluster_colors = [cluster_cmap(i/max(len(unique_clusters)-1,1)) for i in range(len(unique_clusters))]
    cluster_map = {cid: cluster_colors[i] for i,cid in enumerate(unique_clusters)}

    if y_true is not None:
        id_cmap = get_color_map(len(unique_true))
        id_colors = [id_cmap(i/max(len(unique_true)-1,1)) for i,uid in enumerate(unique_true)]
        id_map = {uid: id_colors[i] for i,uid in enumerate(unique_true)}

    def _plot(embedding, name_suffix):
        embedding = np.asarray(embedding)
        fig, axes = plt.subplots(1,2,figsize=(18,8))

        ## Cluster plots
        ax = axes[0]
        for i,(x,y) in enumerate(embedding):
            ax.scatter(x,y,color=cluster_map[y_pred[i]],s=50,alpha=0.7)
        for cid in unique_clusters:
            points = embedding[y_pred==cid]
            if len(points)>1:
                mean = points.mean(axis=0)
                cov = np.cov(points.T)
                vals, vecs = np.linalg.eigh(cov)
                width,height = 2*np.sqrt(np.clip(vals,1e-8,None))
                angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                ax.add_patch(Ellipse(xy=mean,width=width,height=height,angle=angle,alpha=0.2,color=cluster_map[cid]))
        legend_elements = []
        for cid in unique_clusters:
            member_ids = list(ids_array[y_pred==cid])
            label = f"Cluster {cid}: "+", ".join(member_ids)
            legend_elements.append(Patch(color=cluster_map[cid],label="\n".join(textwrap.wrap(label,wrap_width))))
        ax.legend(handles=legend_elements,bbox_to_anchor=(1.05,1),loc='upper left',fontsize=8,frameon=False)
        ax.set_title(f"VaDE Clusters ({tag})"); ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2"); ax.grid(False)

        ## True IDs
        if y_true is not None:
            ax = axes[1]
            for i,(x,y) in enumerate(embedding):
                ax.scatter(x,y,color=id_map[y_true_enc[i]],s=50,alpha=0.7)
            legend_elements = []
            for uid in unique_true:
                points = embedding[y_true_enc==uid]
                if len(points)>1:
                    mean = points.mean(axis=0)
                    cov = np.cov(points.T)
                    vals, vecs = np.linalg.eigh(cov)
                    width,height = 2*np.sqrt(np.clip(vals,1e-8,None))
                    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                    ax.add_patch(Ellipse(xy=mean,width=width,height=height,angle=angle,alpha=0.2,color=id_map[uid]))
                member_ids = list(ids_array[y_true_enc==uid])
                label = f"ID {le_local.inverse_transform([uid])[0]}: "+", ".join(member_ids)
                legend_elements.append(Patch(color=id_map[uid],label="\n".join(textwrap.wrap(label,wrap_width))))
            ax.legend(handles=legend_elements,bbox_to_anchor=(1.05,1),loc='upper left',fontsize=8,frameon=False)
            ax.set_title(f"True IDs ({tag})"); ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2"); ax.grid(False)

        plt.tight_layout()
        outname = os.path.join(args.plots_dir,f"{name_suffix}_{tag}.pdf")
        plt.savefig(outname,bbox_inches='tight')
        plt.close()
        print(f"Written plot to: {outname}")

        ## Export embedding
        emb_df = pd.DataFrame(embedding, columns=["x1","x2"])
        emb_df["cluster"] = y_pred
        if y_true is not None:
            emb_df["ID"] = y_true
        csv_path = os.path.join(args.plots_dir,f"{name_suffix}_{tag}_points.csv")
        emb_df.to_csv(csv_path, index=False)
        print(f"Embedding CSV written to: {csv_path}")

    latent_clean = np.nan_to_num(latent, nan=0.0, posinf=1e6, neginf=-1e6)
    valid_mask = np.isfinite(latent_clean).all(axis=1)
    latent_clean = latent_clean[valid_mask]
    y_pred_clean = np.array(y_pred)[valid_mask]
    ids_clean = np.array(ids_array)[valid_mask]
    y_true_clean = np.array(y_true)[valid_mask] if y_true is not None else None
    print(f"Cleaned latent: {latent_clean.shape[0]} valid samples out of {latent.shape[0]}")

    ## PCA
    try:
        pca_emb = PCA(n_components=2, random_state=1225).fit_transform(latent_clean)
        _plot(pca_emb,"latent_pca_combined_ellipses")
    except Exception as e:
        print(f"[Warning] PCA failed for {tag}: {e}")

    ## t-SNE
    try:
        perplexity = min(30, max(2, latent_clean.shape[0]//3))
        tsne_emb = TSNE(n_components=2, random_state=1225, perplexity=perplexity).fit_transform(latent_clean)
        _plot(tsne_emb,"latent_tsne_combined_ellipses")
    except Exception as e:
        print(f"[Warning] t-SNE failed for {tag}: {e}. Skipping t-SNE plot.")

## Extract full clusters
dataset_full = TensorDataset(torch.from_numpy(X.astype(np.float32)))
latent_full, clusters_full = extract_clusters(model, dataset_full, batch_size=args.batch_size)

## Export full latent CSV
export_df = pd.DataFrame(latent_full, columns=[f"latent_{i+1}" for i in range(latent_full.shape[1])])
export_df["cluster"] = clusters_full
export_df["ID"] = ids
export_path = os.path.join(args.plots_dir, "latent_full_data.csv")
export_df.to_csv(export_path, index=False)
print(f"Full-model latent data written to: {export_path}")

## Plot full model embeddings
plot_embedding_combined(latent_full, ids, clusters_full, y_true=ids, tag="full_model")

## Soft cluster assignment
def plot_soft_assignments(model, dataset, ids_array, outpath, csv_outpath):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    q_list = []

    ## Compute probs
    with torch.no_grad():
        for xb_tuple in loader:
            xb = xb_tuple[0].to(device)

            mu_z, logvar_z = model.encoder(xb)
            z = mu_z 

            z_exp = z.unsqueeze(1)                          
            mu_c = model.mu_c.unsqueeze(0)                  
            logvar_c = model.logvar_c.unsqueeze(0)        

            logvar_c = torch.clamp(logvar_c, min=-6.0, max=6.0)

            log_p_z_given_c = -0.5 * (
                torch.sum(
                    logvar_c + (z_exp - mu_c) ** 2 / torch.exp(logvar_c),
                    dim=2
                )
                + args.latent_dim * math.log(2 * math.pi)
            )

            log_pi = torch.log_softmax(model.pi_logit, dim=0).unsqueeze(0)
            log_pzc = log_pi + log_p_z_given_c

            ## Posterior propbs 
            log_q_c = log_pzc - torch.logsumexp(log_pzc, dim=1, keepdim=True)
            q_c = torch.exp(log_q_c)

            q_list.append(q_c.cpu().numpy())

    ## Stack all samples
    q_all = np.vstack(q_list)

    ## Write CSV outpus
    df_q = pd.DataFrame(
        q_all,
        columns=[f"Cluster_{k}" for k in range(q_all.shape[1])]
    )
    df_q.insert(0, "ID", ids_array)
    df_q.to_csv(csv_outpath, index=False)

    ## Plot bar chart
    sort_idx = np.argsort(np.argmax(q_all, axis=1))
    q_sorted = q_all[sort_idx]
    ids_sorted = np.array(ids_array)[sort_idx]

    n_clusters = q_all.shape[1]
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, n_clusters))

    fig_width = max(16, len(ids_sorted) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bottom = np.zeros(len(ids_sorted))
    for k in range(n_clusters):
        ax.bar(
            np.arange(len(ids_sorted)),
            q_sorted[:, k],
            bottom=bottom,
            color=colors[k],
            label=f"Cluster {k}",
            linewidth=0
        )
        bottom += q_sorted[:, k]

    ax.set_xticks(np.arange(len(ids_sorted)))
    ax.set_xticklabels(ids_sorted, rotation=90, fontsize=12)
    ax.set_ylim(0, 1.05)

    ax.set_xlabel("Samples", fontsize=15)
    ax.set_ylabel("Posterior probability", fontsize=15)
    ax.set_title("VaDE Soft Cluster Assignments", fontsize=18)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="Clusters", fontsize=15)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    print(f"Soft assignments plot written to: {outpath}")
    print(f"Soft assignments CSV written to: {csv_outpath}")

## Plot bar charts
plot_soft_assignments(model, dataset_full, ids,
    os.path.join(args.plots_dir, "soft_assignments_stacked.pdf"),
    os.path.join(args.plots_dir, "soft_assignments.csv")
)

## Ablation
mod_cols_list = df.drop(columns=[args.id_col]).columns.tolist()
modalities_full = {}
for mod_prefix in sorted(list(set([c.split("_")[0] for c in mod_cols_list]))):
    cols = [i for i,c in enumerate(mod_cols_list) if c.startswith(mod_prefix+"_")]
    modalities_full[mod_prefix] = {"cols":cols}

ablation_sets = []
mods = list(modalities_full.keys())
ablation_sets.append(tuple(mods))
for c in itertools.combinations(mods,2):
    ablation_sets.append(c)
for m in mods:
    ablation_sets.append((m,))

X_full_array = np.array(df.drop(columns=[args.id_col]))

## Ablation studies with full pretrain + VaDE training and loss logging
summary_rows = []
summary_rows.append({"ablation":"full","ari_vs_full":1.0,"nmi_vs_full":1.0,
                     "pretrain_loss_csv":"na","train_loss_csv":"na"})

for ab_set in ablation_sets[1:]:
    tag = "_".join(ab_set)
    print(f"\nStarting ablation: {tag}")
    
    ## Prepare ablation dataset
    X_ab = np.zeros_like(X_full_array)
    for m in ab_set:
        X_ab[:, modalities_full[m]['cols']] = X_full_array[:, modalities_full[m]['cols']]
    dataset_ab = TensorDataset(torch.from_numpy(X_ab.astype(np.float32)))
    loader_ab = DataLoader(dataset_ab, batch_size=args.batch_size, shuffle=True)
    
    ## Initialize ablation model
    ab_model = VaDE(input_dim, args.hidden_dims, args.latent_dim, best_k).to(device)
    
    ## Pretrain encoder + decoder
    ab_pretrain_optimizer = torch.optim.Adam(list(ab_model.encoder.parameters()) + list(ab_model.decoder.parameters()), lr=args.pretrain_lr)
    ab_pretrain_loss_history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        ab_model.train()
        total_loss = 0.0
        for xb_tuple in loader_ab:
            xb = xb_tuple[0].to(device)
            xr, z, mu, logvar = ab_model(xb)
            recon = F.binary_cross_entropy_with_logits(xr, xb, reduction='sum')/xb.size(0) if args.is_binary else F.mse_loss(xr, xb, reduction='sum')/xb.size(0)
            kl = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/xb.size(0)
            loss = recon + kl
            ab_pretrain_optimizer.zero_grad()
            loss.backward()
            ab_pretrain_optimizer.step()
            total_loss += loss.item()
        ab_pretrain_loss_history.append(total_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Ablation {tag} Pretrain] Epoch {epoch}: Loss {total_loss:.6f}")
    
    ## Write pretrain loss
    ab_pretrain_csv = os.path.join(args.plots_dir, f"pretrain_loss_history_{tag}.csv")
    pd.DataFrame({"epoch": np.arange(1,args.pretrain_epochs+1), "loss": ab_pretrain_loss_history}).to_csv(ab_pretrain_csv, index=False)
    print(f"Ablation {tag} pretrain loss written to: {ab_pretrain_csv}")
    
    ## Initialize GMM parameters from full model
    with torch.no_grad():
        ab_model.pi_logit.copy_(model.pi_logit)
        ab_model.mu_c.copy_(model.mu_c)
        ab_model.logvar_c.copy_(model.logvar_c)
    
    ## VaDE training
    ab_train_optimizer = torch.optim.Adam(ab_model.parameters(), lr=args.vade_lr)
    ab_train_loss_history = []
    for epoch in range(1, args.vade_epochs + 1):
        ab_model.train()
        total_loss = 0.0
        beta = min(1.0, epoch / args.beta_anneal_epochs)
        for xb_tuple in loader_ab:
            xb = xb_tuple[0].to(device)
            xr, z, mu, logvar = ab_model(xb)
            batch = xb.size(0)
            recon_loss = F.binary_cross_entropy_with_logits(xr, xb, reduction="sum")/batch if args.is_binary else F.mse_loss(xr, xb, reduction="sum")/batch
            z_exp = z.unsqueeze(1)
            mu_c = ab_model.mu_c.unsqueeze(0)
            logvar_c = ab_model.logvar_c.unsqueeze(0)
            log_p_z_given_c = -0.5*(torch.sum(logvar_c + (z_exp - mu_c)**2 / torch.exp(logvar_c), dim=2) + args.latent_dim*math.log(2*math.pi))
            log_pi = torch.log_softmax(ab_model.pi_logit, dim=0).unsqueeze(0)
            log_pzc = log_pi + log_p_z_given_c
            log_pz = torch.logsumexp(log_pzc, dim=1)
            const_term = args.latent_dim*(1+math.log(2*math.pi))
            E_log_q = -0.5*(const_term + torch.sum(logvar, dim=1))
            kl_per_sample = E_log_q - log_pz
            kl_term = torch.mean(kl_per_sample)
            loss_batch = recon_loss + beta*kl_term
            ab_train_optimizer.zero_grad()
            loss_batch.backward()
            ab_train_optimizer.step()
            total_loss += loss_batch.item()
        ab_train_loss_history.append(total_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Ablation {tag} VaDE] Epoch {epoch}: Loss {total_loss:.6f} (β={beta:.2f})")
    
    ## Write VaDE training loss
    ab_train_csv = os.path.join(args.plots_dir, f"vade_loss_history_{tag}.csv")
    pd.DataFrame({"epoch": np.arange(1,args.vade_epochs+1), "loss": ab_train_loss_history}).to_csv(ab_train_csv, index=False)
    print(f"Ablation {tag} VaDE training loss written to: {ab_train_csv}")
    
    ## Compute clusters after fine-tuning
    loader_ab_eval = DataLoader(dataset_ab, batch_size=args.batch_size, shuffle=False)
    latent_ab, clusters_ab = compute_clusters(ab_model, loader_ab_eval)
    
    ## Clean latent
    latent_ab = np.nan_to_num(latent_ab, nan=0.0, posinf=1e6, neginf=-1e6)
    valid_mask = np.isfinite(latent_ab).all(axis=1)
    latent_ab = latent_ab[valid_mask]
    clusters_ab = clusters_ab[valid_mask]
    ids_clean = np.array(ids)[valid_mask]
    
    # Write ablation latent
    export_df_ab = pd.DataFrame(latent_ab, columns=[f"latent_{i+1}" for i in range(latent_ab.shape[1])])
    export_df_ab["cluster"] = clusters_ab
    export_df_ab["ID"] = ids_clean
    export_path_ab = os.path.join(args.plots_dir, f"latent_ablation_{tag}_data.csv")
    export_df_ab.to_csv(export_path_ab, index=False)
    print(f"Ablation latent data written to: {export_path_ab}")
    
    ## Plot embedding
    plot_embedding_combined(latent_ab, ids_clean, clusters_ab, y_true=ids_clean, tag=f"ablation_{tag}")
    
    ## ARI and NMI vs full model
    ari = adjusted_rand_score(clusters_full, clusters_ab)
    nmi = normalized_mutual_info_score(clusters_full, clusters_ab)
    summary_rows.append({
        "ablation": tag,
        "ari_vs_full": ari,
        "nmi_vs_full": nmi,
        "pretrain_loss_csv": ab_pretrain_csv,
        "train_loss_csv": ab_train_csv
    })
    print(f"Ablation {tag}: ARI={ari:.4f}, NMI={nmi:.4f}")

## Write ablation summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(args.plots_dir, "ablation_summary.csv"), index=False)
print("All ablations done, summary saved.")


## ARI and NMI bar charts
def save_ari_nmi_barchart(summary_df, outpath):
    labels = summary_df['ablation'].tolist()
    ari_vals = summary_df['ari_vs_full'].tolist()
    nmi_vals = summary_df['nmi_vs_full'].tolist()
    x = np.arange(len(labels))
    width = 0.6
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    
    ## ARI
    ax = axes[0]
    ax.bar(x, ari_vals, width, edgecolor='black')
    ax.set_ylim(0,1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title("ARI vs Full Model")
    ax.set_ylabel("Adjusted Rand Index")
    for i, v in enumerate(ari_vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    
    ## NMI
    ax = axes[1]
    ax.bar(x, nmi_vals, width, edgecolor='black')
    ax.set_ylim(0,1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title("NMI vs Full Model")
    ax.set_ylabel("Normalized Mutual Information")
    for i, v in enumerate(nmi_vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"ARI and NMI bar charts written to {outpath}")

save_ari_nmi_barchart(summary_df, os.path.join(args.plots_dir, "ablation_ari_nmi_barchart.pdf"))

print(f"All outputs written to: {args.plots_dir}")
