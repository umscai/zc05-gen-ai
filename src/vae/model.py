import logging
import torch
import torch.nn as nn
from typing import Any

logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """A convolutional block: Conv -> BatchNorm (optional) -> ReLU/LeakyReLU.
    Args:
        - in_channels: Number of input channels.
        - out_channels: Number of output feature maps.
        - cfg: Configuration object containing other parameters (e.g. batch_norm (apply or not) or negative_slope of LeakyReLU (zero means ReLU)
    """
    def __init__(self, in_channels: int, out_channels: int, cfg: Any):
        super().__init__()
        # when bias is not given it set to True, [TODO] We may set to False if Conv2d follow by a BatchNorm2d
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride=2, padding=1)
        self.batch_norm = cfg.batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        if cfg.negative_slope >0:
            self.activation = nn.LeakyReLU(cfg.negative_slope) # default negative_slope = 1e-2
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x

#----------------------------------------------------------------------------------------
class DeconvBlock(nn.Module):
    """A transposed convolutional block: ConvTranspose -> BatchNorm (optional) -> ReLU/LeakyReLU.
    Args:
        - in_channels: Number of input channels.
        - out_channels: Number of output feature maps.
        - cfg: Configuration object containing other parameters (e.g. batch_norm (apply or not) or negative_slope of LeakyReLU (zero means ReLU)
    """
    def __init__(self, in_channels: int, out_channels: int, cfg: Any):
        super().__init__()
        # when bias is not given it set to True, [TODO] We may set to False if Conv2d follow by a BatchNorm2d
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding =1)
        self.batch_norm = cfg.batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        if cfg.negative_slope >0:
            self.activation = nn.LeakyReLU(cfg.negative_slope) # default negative_slope = 1e-2
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x

#----------------------------------------------------------------------------------------
class Encoder(nn.Module):
    """Convolutional Encoder (Recognition/Inference network) for VAE.
    Maps input image x to latent distribution parameters (mu, logvar).
    Args:
        - cfg: Configuration object containing required parameter:
            latent_dim: Dimension of latent Gaussian distribution z.
            conv_dim: Base feature channels count for initial layer (default: 64).
            num_conv_layer: Total number of downsampling conv layers (default: 4).
            image_size: Input spatial height/width (default: 64).
    """
    def __init__(self, cfg:Any):
        super().__init__()

        # current values
        in_channel = 3  # RGB image input
        out_channel = cfg.conv_dim # number of filters in initial layers
        spatial_dim = cfg.image_size

        conv_layer_list = []
        for _ in range(cfg.num_conv_layer):
            conv_layer_list.append(ConvBlock(in_channel, out_channel, cfg))
            # Each ConvBlock with stride=2 halves the dimension
            spatial_dim = spatial_dim // 2
            in_channel = out_channel
            out_channel *= 2

        self.conv_layers = nn.Sequential(*conv_layer_list)

        # Flattened dimension after spatial reduction
        # assume we start with RGB-3-channel 64x64 image, and we have 4 layers
        # I: 3, 64x64--> O1: 64, 32x32, O2: 128, 16x16, O3: 256, 8x8, O4: 512, 4x4
        # at the end of loop we have in_channel CH with spatial_dim
        self.final_channels = in_channel
        self.spatial_dim = spatial_dim
        self.flatten_dim = self.final_channels * self.spatial_dim * self.spatial_dim

        # Project flattened features to Gaussian latent parameters: Mean (mu) and Log-Variance (logvar)
        self.fc_mu = nn.Linear(self.flatten_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1) # keep-batch, flatten the rest, note torch.nn.Flatten() keep batch automatically (has start_dim=1 by default but torch.flatten default start_dim=0)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

#----------------------------------------------------------------------------------------
class Decoder(nn.Module):
    """Deconvolutional Decoder (Generator) for VAE. (Image Generator from latent)
    Maps latent representation z back to reconstructed image space.
    Args:
        - cfg: Configuration object containing required parameter:
            latent_dim: Dimension of latent Gaussian distribution z.
            conv_dim: Base feature channels count for initial layer (default: 64).
            num_conv_layer: Total number of downsampling conv layers (default: 4).
            image_size: Input spatial height/width (default: 64).
            last_activation_sigmoid: if True--> sigmoid, we use Sigmoid when image [0,1] and Tanh when imahe [-1,1]
    """
    def __init__(self, cfg:Any):
        super().__init__()
        self.spatial_dim = cfg.image_size // (2**cfg.num_conv_layer)
        self.init_channels = (2 ** (cfg.num_conv_layer - 1)) * cfg.conv_dim
        self.flatten_dim = self.init_channels * self.spatial_dim * self.spatial_dim

        # Project 1D latent z to 4D initial feature volume
        self.fc_decoder = nn.Linear(cfg.latent_dim, self.flatten_dim)
        in_channel = self.init_channels
        out_channel = in_channel // 2
        deconv_layer_list = []

        for j in range(cfg.num_conv_layer - 1):
            deconv_layer_list.append(DeconvBlock(in_channel, out_channel, cfg))
            in_channel = out_channel
            out_channel = out_channel // 2

        # Final reconstruction deconv block (upsamples to output image dimensions) no ReLU or BatchNorm, only ConvTranspose2d layer
        deconv_layer_list.append(nn.ConvTranspose2d(in_channel, 3, kernel_size=4, stride=2, padding=1))

        self.deconv_layers = nn.Sequential(*deconv_layer_list)
        # Sigmoid for pixel intensities normalized in range [0, 1], [-1,1] if normalized to [-1,1]
        if cfg.last_activation_sigmoid:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decoder(z)
        # Reshape to 4D tensor: [Batch, Channels, Height, Width]
        x = x.view(-1, self.init_channels, self.spatial_dim, self.spatial_dim)
        x = self.deconv_layers(x)
        x = self.last_activation(x)
        return x

#----------------------------------------------------------------------------------------
class VAE(nn.Module):
    """Complete Variational Autoencoder wrapping Encoder, Reparameterization, and Decoder.
    Args:
        - cfg: Configuration object containing required parameters, e.g.:
            latent_dim: Dimension of latent vector z.
            conv_dim: Base feature channel count (default: 64).
            num_conv_layer: Total downsampling/upsampling layers (default: 4).
            image_size: Input and output spatial resolution (default: 64).
    """
    def __init__(self, cfg:Any):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z = mu + std * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of VAE.
        Returns:
            recon_x: Reconstructed output image tensor [Batch, 3, H, W].
            mu: Latent distribution mean tensor [Batch, latent_dim].
            logvar: Latent distribution log-variance tensor [Batch, latent_dim].
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

