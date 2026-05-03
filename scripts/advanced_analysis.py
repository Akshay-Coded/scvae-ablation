import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from data.dataset import prepare_pbmc_data
from models.vae import scRNA_VAE


def load_model_and_extract(model_path, loss_type, adata, device):
    model = scRNA_VAE(input_dim=2000, loss_type=loss_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    x_norm = torch.tensor(x_matrix, dtype=torch.float32).to(device)

    # Calculate library size directly from the matrix
    lib_size_array = x_matrix.sum(axis=1)
    lib_size = torch.tensor(lib_size_array, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        h_enc = model.encoder(x_norm)
        mu_z = model.fc_mu(h_enc)
        outputs = model(x_norm, library_size=lib_size)

        # FIX: Check the loss_type to pull the correct reconstruction key!
        if loss_type == "mse":
            recon = outputs["recon_x"].cpu().numpy()
        else:
            recon = outputs["mu_recon"].cpu().numpy()

    return mu_z.cpu().numpy(), recon, x_matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- Phase 9: Advanced Metrics ---")

    adata = prepare_pbmc_data(n_top_genes=2000)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
    biological_labels = adata.obs["leiden"].values
    n_clusters = len(np.unique(biological_labels))

    models = ["mse", "nb", "zinb"]
    ari_scores = {}

    # ==========================================
    # GRAPH 1: Adjusted Rand Index (ARI)
    # ==========================================
    print("Calculating ARI scores...")
    for m in models:
        path = f"checkpoints/best_model_{m}.pt"
        z, _, _ = load_model_and_extract(path, m, adata, device)

        # Run standard KMeans on the latent space to see what the VAE "thinks" the clusters are
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        vae_labels = kmeans.fit_predict(z)

        score = adjusted_rand_score(biological_labels, vae_labels)
        ari_scores[m.upper()] = score

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=list(ari_scores.keys()),
        y=list(ari_scores.values()),
        palette=["#ffb3e6", "#c2c2f0", "#ffcc99"],
    )
    plt.ylabel("Adjusted Rand Index (ARI)", fontsize=12)
    plt.title("Biological Accuracy of Latent Spaces", fontsize=14)
    for i, v in enumerate(ari_scores.values()):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig("report_ari_scores.png", dpi=300)
    plt.close()

    # ==========================================
    # GRAPH 2 & 3: ZINB Specific Deep-Dives
    # ==========================================
    print("Generating ZINB Heatmap and Reconstruction plot...")
    z_zinb, recon_zinb, raw_matrix = load_model_and_extract(
        "checkpoints/best_model_zinb.pt", "zinb", adata, device
    )

    # Graph 2: Cluster Distance Heatmap
    unique_labels = np.unique(biological_labels)
    cluster_centers = []
    for label in unique_labels:
        cluster_centers.append(np.mean(z_zinb[biological_labels == label], axis=0))

    dist_matrix = squareform(pdist(cluster_centers, metric="euclidean"))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        dist_matrix,
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cmap="viridis",
        annot=False,
    )
    plt.xlabel("Cluster ID")
    plt.ylabel("Cluster ID")
    plt.title("ZINB Latent Space: Euclidean Distance Between Clusters")
    plt.tight_layout()
    plt.savefig("report_zinb_distance_heatmap.png", dpi=300)
    plt.close()

    # Graph 3: Reconstructed vs Raw Mean Expression
    real_mean = np.mean(raw_matrix, axis=0)
    fake_mean = np.mean(recon_zinb, axis=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(real_mean, fake_mean, alpha=0.3, color="indigo", s=10)
    # Plot a perfect diagonal line for reference
    max_val = max(np.max(real_mean), np.max(fake_mean))
    plt.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Reconstruction")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("True Mean Gene Expression", fontsize=12)
    plt.ylabel("ZINB Predicted Mean Gene Expression", fontsize=12)
    plt.title("Generative Performance: True vs Predicted", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("report_reconstruction_quality.png", dpi=300)
    plt.close()

    print(
        "Success! Saved 'report_ari_scores.png', 'report_zinb_distance_heatmap.png', and 'report_reconstruction_quality.png'."
    )


if __name__ == "__main__":
    main()
