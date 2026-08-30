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



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# # ----------------------------------------------------------------------------------------
# class ConvBlock(nn.Module):
#     """A convolutional block: Conv -> BatchNorm (optional) -> LeakyReLU.

#     Args:
#         in_channels: Number of input channels.
#         out_channels: Number of output feature maps.
#         kernel_size: Convolution filter dimension (default = 4).
#         batch_norm: Whether to apply batch normalization (default = True).
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int = 4,
#         batch_norm: bool = True,
#     ):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=2,
#             padding=1,
#             #bias=False, #[TODO] make it dynamic if follow by batchnorm=False, otherwise True
#         )
#         self.batch_norm = batch_norm
#         if self.batch_norm:
#             self.bn = nn.BatchNorm2d(out_channels)
#         self.activation = nn.LeakyReLU() # default negative_slope = 1e-2

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv(x)
#         if self.batch_norm:
#             x = self.bn(x)
#         x = self.activation(x)
#         return x


# # ----------------------------------------------------------------------------------------
# class DeconvBlock(nn.Module):
#     """A transposed convolutional block: ConvTranspose -> BatchNorm (optional) -> ReLU.

#     Args:
#         in_channels: Number of input channels.
#         out_channels: Number of output feature maps.
#         kernel_size: Transposed conv filter dimension (default = 4).
#         stride: Stride of transposed conv (default = 2).
#         padding: Padding of transposed conv (default = 1).
#         batch_norm: Whether to apply batch normalization (default = True).
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int = 4,
#         stride: int = 2,
#         padding: int = 1,
#         batch_norm: bool = True,
#     ):
#         super(DeconvBlock, self).__init__()
#         self.deconv = nn.ConvTranspose2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             padding,
#             #bias=False, #[TODO] make it dynamic if follow by batchnorm=False, otherwise True
#         )
#         self.batch_norm = batch_norm
#         if self.batch_norm:
#             self.bn = nn.BatchNorm2d(out_channels)
#         self.activation = nn.LeakyReLU() # nn.ReLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.deconv(x)
#         if self.batch_norm:
#             x = self.bn(x)
#         x = self.activation(x)
#         return x


# # ----------------------------------------------------------------------------------------
# class Encoder(nn.Module):
#     """Convolutional Encoder for VAE.

#     Maps input image x to latent distribution parameters (mu, logvar).

#     Args:
#         latent_dim: Dimension of latent Gaussian distribution z.
#         conv_dim: Base feature channels count for initial layer (default: 64).
#         num_conv_layer: Total number of downsampling conv layers (default: 4).
#         image_size: Input spatial height/width (default: 64).
#     """

#     def __init__(
#         self,
#         latent_dim: int = 128,
#         conv_dim: int = 64,
#         num_conv_layer: int = 4,
#         image_size: int = 64,
#     ):
#         super(Encoder, self).__init__()

#         in_channel = 3  # RGB image input
#         out_channel = conv_dim # number of filters in initial layers
#         batch_norm_ = True # if False  # Skip BatchNorm on first layer (similar to DCGAN guidelines)
#         conv_layer_list = []
#         # Track the spatial dimension (Height/Width), set for our dataset 64
#         spatial_dim = image_size

#         for _ in range(num_conv_layer):
#             # Each ConvBlock with stride=2 halves the dimension
#             spatial_dim = spatial_dim // 2
#             conv_layer_list.append(
#                 ConvBlock(in_channel, out_channel, batch_norm=batch_norm_)
#             )
#             batch_norm_ = True  # Enable BatchNorm for all subsequent layers
#             in_channel = out_channel
#             out_channel *= 2

#         self.conv_layers = nn.Sequential(*conv_layer_list)

#         # Flattened dimension after spatial reduction
#         # assume we start with RGB-3-channel 64x64 image, and we have 4 layers
#         # I: 3, 64x64--> O1: 64, 32x32, O2: 128, 16x16, O3: 256, 8x8, O4: 512, 4x4
#         # at the end of loop we have in_channel CH with spatial_dim
#         self.final_channels = in_channel
#         self.spatial_dim = spatial_dim
#         self.flatten_dim = self.final_channels * self.spatial_dim * self.spatial_dim

#         # Project flattened features to Gaussian latent parameters: Mean (mu) and Log-Variance (logvar)
#         self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
#         self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         x = self.conv_layers(x)
#         x = torch.flatten(x, start_dim=1)
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, logvar


# # ----------------------------------------------------------------------------------------
# class Decoder(nn.Module):
#     """Deconvolutional Decoder for VAE. (Image Generator from latent)

#     Maps latent representation z back to reconstructed image space.

#     Args:
#         latent_dim: Dimension of latent vector z.
#         conv_dim: Base feature channels count (default: 64).
#         num_conv_layer: Total number of upsampling deconv layers (default: 4).
#         image_size: Output spatial height/width (default: 64).
#     """

#     def __init__(
#         self,
#         latent_dim: int = 128,
#         conv_dim: int = 64,
#         num_conv_layer: int = 4,
#         image_size: int = 64,
#     ):
#         super(Decoder, self).__init__()

#         self.spatial_dim = image_size // (2**num_conv_layer)
#         self.init_channels = (2 ** (num_conv_layer - 1)) * conv_dim
#         self.flatten_dim = self.init_channels * self.spatial_dim * self.spatial_dim

#         # Project 1D latent z to 4D initial feature volume
#         self.fc_decoder = nn.Linear(latent_dim, self.flatten_dim)

#         in_channel = self.init_channels
#         out_channel = in_channel // 2
#         deconv_layer_list = []

#         for j in range(num_conv_layer - 1):
#             deconv_layer_list.append(
#                 DeconvBlock(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
#             )
#             in_channel = out_channel
#             out_channel = out_channel // 2

#         # Final reconstruction deconv block (upsamples to output image dimensions)
#         deconv_layer_list.append(
#             nn.ConvTranspose2d(in_channel, 3, kernel_size=4, stride=2, padding=1)
#         )

#         self.deconv_layers = nn.Sequential(*deconv_layer_list)
#         # Sigmoid for pixel intensities normalized in range [0, 1], [-1,1] if normalized to [-1,1]
#         self.last_activation = nn.Sigmoid() # nn.Tanh() #

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         x = self.fc_decoder(z)
#         # Reshape to 4D tensor: [Batch, Channels, Height, Width]
#         x = x.view(-1, self.init_channels, self.spatial_dim, self.spatial_dim)
#         x = self.deconv_layers(x)
#         x = self.last_activation(x)
#         return x


# # ----------------------------------------------------------------------------------------
# class VAE(nn.Module):
#     """Complete Variational Autoencoder wrapping Encoder, Reparameterization, and Decoder.

#     Args:
#         latent_dim: Dimension of latent vector z.
#         conv_dim: Base feature channel count (default: 64).
#         num_conv_layer: Total downsampling/upsampling layers (default: 4).
#         image_size: Input and output spatial resolution (default: 64).
#     """

#     def __init__(
#         self,
#         latent_dim: int = 128,
#         conv_dim: int = 64,
#         num_conv_layer: int = 4,
#         image_size: int = 64,
#     ):
#         super(VAE, self).__init__()

#         self.encoder = Encoder(
#             latent_dim=latent_dim,
#             conv_dim=conv_dim,
#             num_conv_layer=num_conv_layer,
#             image_size=image_size,
#         )
#         self.decoder = Decoder(
#             latent_dim=latent_dim,
#             conv_dim=conv_dim,
#             num_conv_layer=num_conv_layer,
#             image_size=image_size,
#         )

#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """Reparameterization trick: sample z = mu + std * epsilon."""
#         #if self.training:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#         # During evaluation mode, deterministically return mean
#         #return mu

#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Forward pass of VAE.

#         Returns:
#             recon_x: Reconstructed output image tensor [Batch, 3, H, W].
#             mu: Latent distribution mean tensor [Batch, latent_dim].
#             logvar: Latent distribution log-variance tensor [Batch, latent_dim].
#         """
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         recon_x = self.decoder(z)
#         return recon_x, mu, logvar

#---------------------------------------------------------------------------------------------------

# import torch
# import logging
# logger = logging.getLogger(__name__)

# #M = 64 # input image MxM
# #F = 64 # number of filters in first conv-layer of discriminator
# #L = 4 # number of conv-layer (max 5)
# #Example with assumption of settings to double the filters, and half the spatial dimension on each layer
# #input = 3 x MxM: 3 x 64x64
# #Layer-1-output: 1F x (M/2 )x(M/2 ):  64 x 32x32
# #Layer-2-output: 2F x (M/4 )x(M/4 ): 128 x 16x16
# #Layer-3-output: 4F x (M/8 )x(M/8 ): 256 x  8x8
# #Layer-4-output: 8F x (M/16)x(M/16): 512 x  4x4
# #So Final output-Channel = 2^(L-1)xF and Spatial = M/2^L
# #and corresponding generator initial layer filters =  2^(L-1)xF=G=8*64=2^3*2*6=2^9
# #using DeconvBlock(latent_dim, G, 4, 1, 0) here the output is exactly the size of kernel


# #----------------------------------------------------------------------------------------
# class ConvBlock(torch.nn.Module):
#     """
#     A convolutional block is made of 3 layers: Conv -> BatchNorm -> Activation.
#     args:
#     - in_channels: number of channels in the input to the conv layer
#     - out_channels: number of filters in the conv layer
#     - kernel_size: filter dimension of the conv layer (default =4)
#     - batch_norm: whether to use batch norm or not (default = True)
#     """
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, batch_norm: bool = True):
#         super(ConvBlock, self).__init__()

#         self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False)
#         self.batch_norm = batch_norm
#         if self.batch_norm:
#             self.bn = torch.nn.BatchNorm2d(out_channels)
#         self.activation = torch.nn.LeakyReLU(0.2)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv(x)
#         if self.batch_norm:
#             x = self.bn(x)
#         x = self.activation(x)
#         return x


# #----------------------------------------------------------------------------------------
# class Discriminator(torch.nn.Module):
#     """
#     The discriminator model
#     args:
#     - conv_dim: control the number of filters in first layer (default: 64)
#     - num_conv_layer: control the number of conv layers (default: 4)
#     - image_size: original input image size for training
#     """
#     def __init__(self, conv_dim: int = 64, num_conv_layer: int = 4, image_size:int=64):
#         super(Discriminator, self).__init__()

#         in_channel = 3 # always start with 3 channels from original RGB image
#         out_channel = conv_dim # number of filters in initial layers
#         batch_norm_ = False # No batch Norm in first layer
#         cov_layer_list = []
#         # Track the spatial dimension (Height/Width), set for our dataset 64
#         output_img_dim = image_size

#         for j in range(num_conv_layer):
#             # Each ConvBlock with stride=2 halves the dimension
#             output_img_dim = output_img_dim // 2
#             cov_layer_list.append(ConvBlock(in_channel, out_channel, batch_norm=batch_norm_)) # kernel=4=default
#             batch_norm_ = True # after first layer is True for the rest
#             in_channel = out_channel # new in_channel = current out_channel
#             out_channel *= 2 # new out_channel is double of the current out_channel


#         #------------------------------
#         #with num_conv_layer = 4: CxWxH, and conv_dim=64
#         #3x 64x64--> 64x 32x32-> 128 x 16x16-> 256 x 8x8 -> 512 x 4 x4
#         #if we used flatten + FC then flatten_size = 64 x 8 X 4x4 = 8192 = 8K
#         #------------------------------
#         self.conv_layers = torch.nn.Sequential(*cov_layer_list)

#         # Instead of Flatten + FC, use a current_dim x current_dim Convolution kernel
#         # This will keep 4D dimension that expected in GAN architecture
#         # otherwise if we use Flatten and FC we end up 2D size [Batch,1] then
#         # we have to add extra x.view(-1, 1, 1, 1), which doesn't sound good!

#         self.final_conv = torch.nn.Conv2d(in_channel, 1, kernel_size=output_img_dim, stride=1, padding=0)
#         #self.sigmoid = nn.Sigmoid() # No-sigmoid when we use BCEWithLogitsLoss for loss which combine BCE and sigmoid for more stable calc


#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.conv_layers(x) # contains all conv layers, e.g. Shape: [Batch, 512, 4, 4]
#         x = self.final_conv(x)   # Shape: [Batch, 1, 1, 1] <--- 4D as expected by GAN
#         #x = self.sigmoid(x) No-sigmoid when we use BCEWithLogitsLoss for loss which combine BCE and sigmoid for more stable calc
#         return x



# #----------------------------------------------------------------------------------------
# class DeconvBlock(torch.nn.Module):
#     """
#     A "de-convolutional" block is made of 3 layers: ConvTranspose -> BatchNorm -> Activation.
#     Args:
#     - in_channels: number of channels in the input to the conv layer
#     - out_channels: number of filters in the conv layer
#     - kernel_size: filter dimension of the conv layer (default = 4)
#     - stride: stride of the conv layer (default = 2)
#     - padding: padding of the conv layer (default = 1)
#     - batch_norm: whether to use batch norm or not (default = True)
#     """
#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 2, padding: int = 1, batch_norm: bool = True):
#         super(DeconvBlock, self).__init__()

#         self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.batch_norm = batch_norm
#         if self.batch_norm:
#             self.bn = torch.nn.BatchNorm2d(out_channels)
#         self.activation = torch.nn.ReLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.deconv(x)
#         if self.batch_norm:
#             x = self.bn(x)
#         x = self.activation(x)
#         return x

# #----------------------------------------------------------------------------------------
# class Generator(torch.nn.Module):
#     """
#     The generator model
#     Args:
#     - latent_dim: dimension of the latent vector (default: 128)
#     - conv_dim: control the number of filters in the conv-transpose layers (default: 64)
#     - num_conv_layer: control the number of conv layers (default: 4)
#     - image_size: input image size (default: 64)
#     """
#     def __init__(self, latent_dim: int = 128, conv_dim: int = 64, num_conv_layer: int = 4, image_size:int=64):
#         super(Generator, self).__init__()

#         in_channel = latent_dim # always start latent_dim
#         out_channel = 2**(num_conv_layer -1)*conv_dim # number of filters in initial layers
#         stride_ = 1 # only for first layer, the rest stride =2
#         padding_ = 0 # only for first layer , the rest padding =1
#         kernel_size_ = image_size//(2**num_conv_layer) # only first layer
#         decov_layer_list = []

#         for j in range(num_conv_layer):
#             decov_layer_list.append(DeconvBlock(in_channel, out_channel, kernel_size=kernel_size_, stride = stride_, padding = padding_)) # kernel=4=default
#             stride_ = 2 # after first layer stride is 2 for the other layers
#             padding_ = 1 # after first layer padding  is 1 for the other layers
#             kernel_size_ = 4
#             in_channel = out_channel     # new in_channel = current out_channel
#             out_channel = out_channel//2 # new out_channel = half of current out_channel
#         #----------------------
#         # add last deconv layer
#         decov_layer_list.append(torch.nn.ConvTranspose2d(in_channel, 3, 4, stride=2, padding=1))
#         #----------------------------------------------------
#         self.deconv_layers = torch.nn.Sequential(*decov_layer_list)
#         #------------------------------------------------------
#         self.last_activation = torch.nn.Tanh()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.deconv_layers(x)
#         x = self.last_activation(x)
#         return x
