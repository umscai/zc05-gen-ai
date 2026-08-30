import logging
import torch
from typing import Any, Tuple


logger = logging.getLogger(__name__)


def get_optimizer(
    model: torch.nn.Module,
    cfg: Any
) -> Tuple[torch.optim.Adam, torch.optim.lr_scheduler.ExponentialLR]:
    """Creates and returns the Adam optimizer for the VAE model.

    Args:
        model: The end-to-end VAE PyTorch model containing both encoder and decoder parameters.
        cfg: Configuration object containing optimizer hyperparameters:
            - lr: Learning rate for the optimizer. Default: 1e-3.
            - beta1: Exponential decay rate for the first moment estimates. Default: 0.9.
            - beta2: Exponential decay rate for the second moment estimates. Default: 0.999.
            - weight_decay: Weight decay (L2 penalty) factor. Default: 0.0.
    Returns:
        torch.optim.Adam: Optimizer configured for all trainable parameters in the VAE.
        and torch.optim.lr_scheduler.ExponentialLR , scheduler to decrease learning rate (we call per epoch)

    Raises:
        RuntimeError: Re-raises any exception caught during optimizer initialization.
    """
    try:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
        return optimizer, scheduler

    except Exception as e:
        logger.error(f"Error initializing VAE optimizer: {e}")
        raise RuntimeError(f"Failed to initialize VAE optimizer: {e}") from e

