import torch
from torch.utils.data import Dataset
import scanpy as sc
import numpy as np


def prepare_pbmc_data(n_top_genes: int = 2000):
    """
    Downloads and preprocesses the PBMC 3k dataset for a generative model.
    Enforces the separation of normalized input (for the encoder)
    and raw counts (for the decoder).
    """
    print("Downloading and loading PBMC 3k dataset...")
    adata = sc.datasets.pbmc3k()

    # 1. Basic Quality Control (QC)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # 2. Extract Library Size BEFORE any normalization
    # This is the exact technical scaling factor 'L' we discussed in Phase 3
    adata.obs["library_size"] = adata.X.sum(axis=1)

    # 3. Save the raw counts in a separate layer
    # We need to keep these intact for the NB/ZINB loss functions
    adata.layers["counts"] = adata.X.copy()

    # 4. Normalize and Log-Transform
    # This stabilizes the variance so the Encoder can learn a smooth manifold
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 5. Highly Variable Gene (HVG) Selection
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)

    print(f"Preprocessing complete. Final shape: {adata.shape}")
    return adata


class scRNADataset(Dataset):
    """
    A strictly typed PyTorch Dataset for single-cell ablation studies.
    """

    def __init__(self, adata):
        # The continuous, scaled data for the Encoder
        self.X_norm = torch.tensor(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
            dtype=torch.float32,
        )

        # The discrete, raw integer counts for the Decoder's likelihood evaluation
        self.X_raw = torch.tensor(
            adata.layers["counts"].toarray()
            if hasattr(adata.layers["counts"], "toarray")
            else adata.layers["counts"],
            dtype=torch.float32,
        )

        # The technical scaling factor (L)
        self.library_size = torch.tensor(
            adata.obs["library_size"].values, dtype=torch.float32
        ).unsqueeze(1)

    def __len__(self):
        return self.X_norm.shape[0]

    def __getitem__(self, idx):
        return {
            "x_norm": self.X_norm[idx],
            "x_raw": self.X_raw[idx],
            "library_size": self.library_size[idx],
        }
