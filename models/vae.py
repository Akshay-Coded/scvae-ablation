import torch
import torch.nn as nn
import torch.nn.functional as F


class scRNA_VAE(nn.Module):
    """
    The upgraded, research-grade VAE for scRNA-seq ablation.
    Includes library size decoupling and modular likelihood heads.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dim: int = 128,
        loss_type: str = "zinb",
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type not in ["mse", "nb", "zinb"]:
            raise ValueError("loss_type must be 'mse', 'nb', or 'zinb'")

        # 1. The Shared Encoder (2 Hidden Layers for sufficient capacity)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout prevents memorization of dropout patterns
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Latent space parameterization
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 2. The Shared Base Decoder
        self.decoder_base = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # 3. Modular Output Heads
        if self.loss_type == "mse":
            # Naive continuous reconstruction
            self.fc_recon = nn.Linear(hidden_dim, input_dim)

        elif self.loss_type in ["nb", "zinb"]:
            # Biological frequency (rho) - heavily relies on Softmax later
            self.fc_rho = nn.Linear(hidden_dim, input_dim)

            # Global dispersion parameter (theta) - 1 per gene
            # Initialized to random normal, mathematically bounded later
            self.theta = nn.Parameter(torch.randn(input_dim))

            if self.loss_type == "zinb":
                # Dropout probability logits (pi)
                self.fc_pi_logits = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Applies the reparameterization trick: z = mu + std * eps"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # At inference time, we just use the mean (no stochasticity)
            return mu

    def forward(self, x: torch.Tensor, library_size: torch.Tensor = None) -> dict:
        """
        Forward pass. Note the inclusion of library_size to scale rho.
        """
        # --- Encoder ---
        h_enc = self.encoder(x)
        mu_z = self.fc_mu(h_enc)
        logvar_z = self.fc_logvar(h_enc)

        # --- Latent Sampling ---
        z = self.reparameterize(mu_z, logvar_z)

        # --- Base Decoder ---
        h_dec = self.decoder_base(z)

        # --- Modular Heads ---
        outputs = {"mu_z": mu_z, "logvar_z": logvar_z}

        if self.loss_type == "mse":
            # Simple continuous prediction
            outputs["recon_x"] = self.fc_recon(h_dec)

        elif self.loss_type in ["nb", "zinb"]:
            # 1. Get relative gene frequencies (rho) summing to 1 per cell
            rho = F.softmax(self.fc_rho(h_dec), dim=-1)

            # 2. Scale by observed technical library size (L * rho)
            # This is the crucial decoupling step for count models
            outputs["mu_recon"] = rho * library_size

            # 3. Apply softplus to dispersion parameter to ensure strict positivity
            outputs["theta"] = F.softplus(self.theta)

            if self.loss_type == "zinb":
                # 4. Get dropout logits (no activation applied here for numerical stability)
                outputs["pi_logits"] = self.fc_pi_logits(h_dec)

        return outputs
