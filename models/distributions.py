import torch
import torch.nn.functional as F


def nb_nll(
    x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes the Negative Log-Likelihood (NLL) for the Negative Binomial distribution.
    All operations are strictly in log-space to prevent underflow/overflow.

    Args:
        x: Raw integer counts (Shape: [Batch, Genes])
        mu: Biological mean scaled by library size (Shape: [Batch, Genes])
        theta: Global dispersion parameter (Shape: [Genes])
        eps: Small constant for numerical stability
    """
    # log(mu + eps) and log(theta + eps) to prevent log(0)
    log_mu = torch.log(mu + eps)
    log_theta = torch.log(theta + eps)
    log_mu_theta = torch.log(mu + theta + eps)

    # 1. The Gamma function terms using torch.lgamma
    # log( Gamma(x + theta) / (Gamma(theta) * Gamma(x + 1)) )
    # = lgamma(x + theta) - lgamma(theta) - lgamma(x + 1)
    # Note: lgamma(x+1) is exactly log(x!)
    term1 = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)

    # 2. The probability terms
    # theta * log(theta / (theta + mu)) + x * log(mu / (theta + mu))
    term2 = theta * (log_theta - log_mu_theta) + x * (log_mu - log_mu_theta)

    # log_p = term1 + term2. We return the NEGATIVE log-likelihood.
    log_p = term1 + term2
    return -log_p


def zinb_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi_logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes the Negative Log-Likelihood (NLL) for the Zero-Inflated Negative Binomial.
    Uses log-sum-exp and logsigmoid for extreme numerical stability.
    """
    # 1. Get the baseline NB log-probability for ALL x
    nb_log_p = -nb_nll(x, mu, theta, eps)  # Re-invert because nb_nll returns negative

    # 2. Compute log(pi) and log(1-pi) stably from logits
    log_pi = F.logsigmoid(pi_logits)
    log_one_minus_pi = F.logsigmoid(-pi_logits)

    # 3. ZINB mixture logic
    # Case A: x == 0
    # log p(x=0) = log( pi + (1-pi) * NB(0) )
    # In log space: torch.logaddexp( log_pi, log_one_minus_pi + nb_log_p(x=0) )
    zero_mask = (x == 0).float()
    log_p_zero = torch.logaddexp(log_pi, log_one_minus_pi + nb_log_p)

    # Case B: x > 0
    # log p(x>0) = log( (1-pi) * NB(x) )
    # In log space: log_one_minus_pi + nb_log_p(x)
    non_zero_mask = (x > 0).float()
    log_p_non_zero = log_one_minus_pi + nb_log_p

    # 4. Combine cases using masks
    log_p = (zero_mask * log_p_zero) + (non_zero_mask * log_p_non_zero)

    return -log_p
