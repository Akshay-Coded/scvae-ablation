import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import scanpy as sc
import matplotlib.pyplot as plt
from data.dataset import prepare_pbmc_data
from models.vae import scRNA_VAE


def extract_latent_space(model_path, loss_type, adata, device):
    """Loads a saved checkpoint and extracts the deterministic latent mean (mu)."""
    # Initialize the specific model variant
    model = scRNA_VAE(input_dim=2000, loss_type=loss_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Toggle evaluation mode (crucial for BatchNorm and deterministic latents)
    model.eval()

    # Extract log-normalized counts (handling both dense and sparse formats)
    x_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    x_norm = torch.tensor(x_matrix, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Pass data through the encoder to get the latent representations
        h_enc = model.encoder(x_norm)
        mu_z = model.fc_mu(h_enc)

    return mu_z.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- Phase 7: Evaluation ---")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading PBMC dataset...")
    adata = prepare_pbmc_data(n_top_genes=2000)

    # Generate baseline biological labels using standard Scanpy pipelines
    # We will use these 'leiden' colors to see how well each VAE groups known cell types
    print("Generating baseline cell type labels...")
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)

    # 2. Extract latent spaces for all 3 models
    models = ["mse", "nb", "zinb"]

    for m in models:
        path = f"checkpoints/best_model_{m}.pt"
        if not os.path.exists(path):
            print(f"Error: {path} not found. Ensure training finished successfully.")
            return

        print(f"Extracting latent space for {m.upper()}...")
        z = extract_latent_space(path, m, adata, device)

        # Store the latent dimensions in the AnnData object
        adata.obsm[f"X_vae_{m}"] = z

        # Compute UMAP neighbors based strictly on the VAE's latent space
        sc.pp.neighbors(adata, use_rep=f"X_vae_{m}", key_added=f"neighbors_{m}")
        sc.tl.umap(adata, neighbors_key=f"neighbors_{m}")

        # Save UMAP coordinates to a specific key so they don't overwrite each other
        adata.obsm[f"X_umap_{m}"] = adata.obsm["X_umap"].copy()

    # 3. Plotting the Ablation Results
    print("Generating comparative UMAP plots...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, m in enumerate(models):
        sc.pl.embedding(
            adata,
            basis=f"X_umap_{m}",
            color="leiden",
            ax=axes[i],
            show=False,
            title=f"{m.upper()} Latent Space",
            legend_loc="none" if i < 2 else "right margin",
        )

    plt.tight_layout()
    plt.savefig("ablation_results.png", dpi=300, bbox_inches="tight")
    print("Success! Open 'ablation_results.png' to view your findings.")


if __name__ == "__main__":
    main()
