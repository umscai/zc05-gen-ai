import torch
#import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
                      logvar: torch.Tensor, kld_weight: float = 1.0, reduction: str = "mean", rec_loss_type:str='MSE') -> dict[str, torch.Tensor]:
    """Calculates the total VAE loss along with its individual component losses.
    Args:
        recon_x: Reconstructed output tensor from the decoder of shape (N, ...) where N=batch-size
        x: Original input tensor of shape (N, ...).
        mu: Mean tensor of the latent distribution of shape (N, D), where D is the latent dimension size.
        logvar: Log-variance tensor of the latent distribution of shape (N, D).
        kld_weight: Scaling factor (beta) applied to the KL divergence loss (Default: 1.0)
        reduction: Specifies the reduction method: 'mean', 'sum', or 'none'. Default: 'mean'.
        rec_loss_type: reconstruction loss type: BCE, L1, MSE, MSEL1

    Returns:
        dict[str, torch.Tensor]: A dictionary containing:
            - 'total_loss': Weighted sum of reconstruction and KLD losses.
            - 'recon_loss': Reconstruction error/loss
            - 'kld_loss': Analytical KL Divergence loss against a standard normal prior.

    Raises:
        ValueError: If `reduction` is not 'mean', 'sum', or 'none'.or Re-raises any exception caught during processing after logging.
    """
    try:
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction mode: '{reduction}'. Supported modes: 'mean', 'sum', 'none'.")

        # NOTE: torch.nn.functional losses with reduction='none' preserve input shape (N, ...).

        # Reconstruction Loss (MSE or BCE, L1, MSEL1)
        if rec_loss_type == 'BCE':
            recon_loss = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction=reduction)
        elif rec_loss_type == 'L1':
            recon_loss = torch.nn.functional.l1_loss(recon_x, x, reduction=reduction)
        elif rec_loss_type == "MSEL1":
            mse_loss = torch.nn.functional.mse_loss(recon_x, x, reduction=reduction)
            l1_loss = torch.nn.functional.l1_loss(recon_x, x, reduction=reduction)

            l1_weight = 0.5  # Adjust between 0.0 and 1.0 depending on desired sharpness (L1 generate sharper image than MSE)
            recon_loss = mse_loss + (l1_weight * l1_loss)
        else: #'MSE'
            recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction=reduction)

        # Handle reduction for MSEL1 if reduction='none' (since we add two tensors, check if they are element-wise)
        # Note: torch.nn.functional losses with reduction='none' preserve input shape (N, ...).

        # 2. Analytical KL Divergence for Gaussian prior:
        # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_element = 1 + logvar - mu.pow(2) - logvar.exp()

        if reduction == "sum":
            kld_loss = -0.5 * torch.sum(kld_element)
        elif reduction == "mean":
            # kld_loss = -0.5 * torch.mean(kld_element) # WARNING: Scale mismatch with PyTorch's MSE mean!
            # Average over both batch size and latent dimension to align with mean MSE
            #1-Sum across latent features (dim=1): Calculates the total KL divergence per sample in the batch.
            #2-Average across batch (torch.mean(...)): Normalizes the batch loss by $N$.
            kld_loss = -0.5 * torch.mean(torch.sum(kld_element, dim=1))
        else:  # reduction == "none"
            # Return KLD per sample in the batch (shape: [N]), matching batch-wise inspection
            kld_loss = -0.5 * torch.sum(kld_element, dim=1)

            # If recon_loss has dimensions (e.g., [N, C, H, W] for images like in our case),
            # we need to sum over all dimension except batch, so one error per sample: match batch size [N]
            # since we have element-wise sum on recon_loss kld_loss
            if recon_loss.ndim > 1:
                recon_loss = torch.sum(recon_loss, dim=tuple(range(1, recon_loss.ndim)))

        # Combined Total Loss
        total_loss = recon_loss + (kld_weight * kld_loss)

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kld_loss": kld_loss,
        }
    except Exception as e:
        logger.error(f"Error in vae_loss_function: {e}")
        raise