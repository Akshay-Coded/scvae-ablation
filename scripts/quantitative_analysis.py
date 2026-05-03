import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from data.dataset import prepare_pbmc_data
from models.vae import scRNA_VAE


def extract_latent_space(model_path, loss_type, adata, device):
    """Utility to quickly grab the latent space."""
    model = scRNA_VAE(input_dim=2000, loss_type=loss_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    x_norm = torch.tensor(x_matrix, dtype=torch.float32).to(device)

    with torch.no_grad():
        h_enc = model.encoder(x_norm)
        mu_z = model.fc_mu(h_enc)
    return mu_z.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- Phase 8: Quantitative Metrics ---")

    # 1. Load Data & Generate Labels
    print("Loading dataset and generating baseline labels...")
    adata = prepare_pbmc_data(n_top_genes=2000)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
    labels = adata.obs["leiden"].values

    # Convert to dense matrix for mathematical analysis
    X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    # ==========================================
    # ANALYSIS 1: Sparsity (Dropout) Profiling
    # ==========================================
    print("Generating Sparsity Profile...")
    mean_expression = np.mean(X_dense, axis=0)
    fraction_zeros = np.mean(X_dense == 0, axis=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(mean_expression, fraction_zeros, alpha=0.5, s=10, color="teal")
    plt.xscale("log")
    plt.xlabel("Mean Gene Expression (Log Scale)", fontsize=12)
    plt.ylabel("Fraction of Zeros (Dropout Rate)", fontsize=12)
    plt.title("PBMC Dataset: Gene Expression vs. Sparsity", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("report_sparsity_profile.png", dpi=300)
    plt.close()

    # ==========================================
    # ANALYSIS 2: Silhouette Scores
    # ==========================================
    print("Calculating Silhouette Scores for latent spaces...")
    models = ["mse", "nb", "zinb"]
    scores = {}

    for m in models:
        path = f"checkpoints/best_model_{m}.pt"
        z = extract_latent_space(path, m, adata, device)
        # Silhouette score requires the latent space and the cluster labels
        score = silhouette_score(z, labels)
        scores[m.upper()] = score
        print(f"  {m.upper()} Silhouette Score: {score:.4f}")

    # Plot the scores
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=list(scores.keys()),
        y=list(scores.values()),
        palette=["#ff9999", "#66b3ff", "#99ff99"],
    )
    plt.ylabel("Silhouette Score (Higher is Better)", fontsize=12)
    plt.title("Clustering Performance across Loss Functions", fontsize=14)
    plt.ylim(0, max(scores.values()) * 1.2)  # Give headroom

    # Add exact numbers on top of bars
    for i, v in enumerate(scores.values()):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig("report_silhouette_scores.png", dpi=300)
    plt.close()

    print(
        "\nSuccess! Saved 'report_sparsity_profile.png' and 'report_silhouette_scores.png'."
    )


if __name__ == "__main__":
    main()
