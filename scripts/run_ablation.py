import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from data.dataset import prepare_pbmc_data, scRNADataset
from models.vae import scRNA_VAE
from models.distributions import nb_nll, zinb_nll


def compute_elbo_loss(outputs, x_raw, beta, loss_type):
    # ... (Keep your existing compute_elbo_loss code exactly as is)
    mu_z = outputs["mu_z"]
    logvar_z = outputs["logvar_z"]
    kl_loss = (
        -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1).mean()
    )

    if loss_type == "mse":
        recon_loss = (
            F.mse_loss(outputs["recon_x"], x_raw, reduction="none").sum(dim=1).mean()
        )
    elif loss_type == "nb":
        recon_loss = (
            nb_nll(x=x_raw, mu=outputs["mu_recon"], theta=outputs["theta"])
            .sum(dim=1)
            .mean()
        )
    elif loss_type == "zinb":
        recon_loss = (
            zinb_nll(
                x=x_raw,
                mu=outputs["mu_recon"],
                theta=outputs["theta"],
                pi_logits=outputs["pi_logits"],
            )
            .sum(dim=1)
            .mean()
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    total_loss = recon_loss + (beta * kl_loss)
    return total_loss, recon_loss, kl_loss


def train_model(
    loss_type: str, epochs: int = 100, batch_size: int = 128, max_beta: float = 1.0
):
    print(f"\n--- Starting Ablation Run: {loss_type.upper()} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    adata = prepare_pbmc_data(n_top_genes=2000)
    dataset = scRNADataset(adata)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = scRNA_VAE(input_dim=2000, loss_type=loss_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # Tracker for saving the best model
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        beta = min(max_beta, (epoch + 1) / 50.0)

        for batch in train_loader:
            x_norm = batch["x_norm"].to(device)
            x_raw = batch["x_raw"].to(device)
            lib_size = batch["library_size"].to(device)

            optimizer.zero_grad()
            outputs = model(x_norm, library_size=lib_size)
            loss, r_loss, kl = compute_elbo_loss(outputs, x_raw, beta, loss_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_norm = batch["x_norm"].to(device)
                x_raw = batch["x_raw"].to(device)
                lib_size = batch["library_size"].to(device)

                outputs = model(x_norm, library_size=lib_size)
                v_loss, _, _ = compute_elbo_loss(outputs, x_raw, beta, loss_type)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # --- Checkpointing Logic ---
        # Only save if the model actually improved
        saved_flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"checkpoints/best_model_{loss_type}.pt")
            saved_flag = " [*Saved Best*]"

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"Epoch [{epoch:3d}/{epochs}] | Beta: {beta:.2f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{saved_flag}"
            )

    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")
    return model, adata


if __name__ == "__main__":
    # Run all three sequentially.
    # Go grab a coffee, and when you get back, your checkpoints/ folder will have 3 perfect .pt files!
    train_model(loss_type="mse")
    train_model(loss_type="nb")
    train_model(loss_type="zinb")
