#!/usr/bin/env python3

import numpy as np
import pandas as pd
from Bio import Phylo
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
import argparse
import os

## OU function
def simulate_ou(tree, n_traits=25, alpha=0.1, sigma=0.1, root_state=None, theta=None):
    terminals = tree.get_terminals()
    ntips = len(terminals)
    nonterminals = tree.get_nonterminals()
    nnodes = len(nonterminals)
    N = ntips + nnodes

    clade_to_index = {}
    idx = 0
    for tip in terminals:
        clade_to_index[tip] = idx
        idx += 1
    for node in nonterminals:
        clade_to_index[node] = idx
        idx += 1

    node_traits = np.full((N, n_traits), np.nan)

    if root_state is None:
        root_state = np.random.normal(0, 1, size=n_traits)
    if theta is None:
        theta = np.random.normal(0, 1, size=n_traits)

    root_node = tree.root
    root_idx = clade_to_index[root_node]
    node_traits[root_idx, :] = root_state

    def evolve(ancestor, ancestor_idx):
        for child in ancestor.clades:
            descendant_idx = clade_to_index[child]
            t = child.branch_length if child.branch_length else 0.0
            x0 = node_traits[ancestor_idx, :]
            exp_alpha_t = np.exp(-alpha * t)
            mean_t = x0 * exp_alpha_t + theta * (1 - exp_alpha_t)
            var_t = (sigma**2) / (2 * alpha) * (1 - np.exp(-2 * alpha * t))
            node_traits[descendant_idx, :] = np.random.normal(mean_t, np.sqrt(var_t))
            if not child.is_terminal():
                evolve(child, descendant_idx)

    evolve(root_node, root_idx)

    traits_tips = node_traits[:ntips, :]
    tip_labels = [tip.name for tip in terminals]
    df = pd.DataFrame(traits_tips, index=tip_labels,
                      columns=[f"Trait_{i+1}" for i in range(n_traits)])
    return df

## Command line arguments
def main():
    parser = argparse.ArgumentParser(description="OU trait simulation on a phylogenetic tree")
    parser.add_argument("--input", "-i", required=True, help="Input Newick tree file")
    parser.add_argument("--output", "-o", required=True, help="Output directory to write simulated traits")
    parser.add_argument("--n_traits", type=int, default=25, help="Number of traits")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha (selection strength)")
    parser.add_argument("--sigma", type=float, default=0.1, help="Sigma (rate of drift)")
    parser.add_argument("--root_state", type=str, default=None, help="Optional root state values (comma-separated)")
    parser.add_argument("--theta", type=str, default=None, help="Optional OU optima θ values (comma-separated)")
    parser.add_argument("--n_simulations", type=int, default=1, help="Number of simulations to average over")
    parser.add_argument("--show_labels", action="store_true", help="Show tip labels on PCA plot")

    args = parser.parse_args()

    print("Arguments:")
    print(vars(args))

    ## Read guide tree
    tree = Phylo.read(args.input, "newick")

    ## Root_state and theta (if provided)
    root_state = np.array([float(x) for x in args.root_state.split(",")]) if args.root_state else None
    theta = np.array([float(x) for x in args.theta.split(",")]) if args.theta else None

    ## Run simulation
    if args.n_simulations == 1:
        traits_df = simulate_ou(tree, n_traits=args.n_traits, alpha=args.alpha,
                                sigma=args.sigma, root_state=root_state, theta=theta)
    else:
        all_traits = []
        for _ in range(args.n_simulations):
            sim_df = simulate_ou(tree, n_traits=args.n_traits, alpha=args.alpha,
                                 sigma=args.sigma, root_state=root_state, theta=theta)
            all_traits.append(sim_df.values)
        avg_traits = np.mean(np.stack(all_traits), axis=0)
        traits_df = pd.DataFrame(avg_traits, index=sim_df.index, columns=sim_df.columns)

    traits_df = traits_df.round(2)

    ## Extract population/group from tip label (for PCA labels)
    traits_df['group'] = [name.split('_')[-1] for name in traits_df.index]

    ## Create output directory
    os.makedirs(args.output, exist_ok=True)

    ## Write traits to CSV
    csv_path = os.path.join(args.output, "tip_traits.csv")
    traits_df.to_csv(csv_path, float_format="%.2f")
    print(f"Tip traits written to {csv_path}")

    ## PCA plot to visualize simulated traits
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(traits_df.iloc[:, :-1].values)
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100

    ## Set population colors
    groups = sorted(traits_df['group'].unique(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    palette_colors = plt.cm.Set2.colors
    if len(groups) > len(palette_colors):
        palette_colors = ListedColormap(plt.cm.tab20.colors)(np.linspace(0,1,len(groups)))
    color_map = {g: palette_colors[i % len(palette_colors)] for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for g in groups:
        mask = traits_df['group'] == g
        ax.scatter(pcs[mask, 0], pcs[mask, 1], label=g, alpha=0.8, color=color_map[g])

    ## Show tip labels (if requested)
    if args.show_labels:
        for i, label in enumerate(traits_df.index):
            ax.text(pcs[i, 0], pcs[i, 1], label, fontsize=8, alpha=0.7)

    ax.set_xlabel(f"PC1 ({pc1_var:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}% var)")
    ax.set_title("PCA of OU Simulated Traits on Tree Tips")
    ax.legend(title="Population", loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()

    ## Write PCA plot
    fig_path = os.path.join(args.output, "pca_plot.png")
    fig.savefig(fig_path, dpi=300)
    print(f"PCA plot written to {fig_path}")


if __name__ == "__main__":
    main()
