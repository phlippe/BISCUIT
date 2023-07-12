import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """ 
    Convolution encoder network 
    We use a stack of convolutions with strides in every second convolution to reduce
    dimensionality. For the datasets in question, the network showed to be sufficient.
    """

    def __init__(self, c_hid, num_latents,
                 c_in=3,
                 width=32,
                 act_fn=lambda: nn.SiLU(),
                 use_batch_norm=False,
                 residual=False,
                 num_blocks=1,
                 variational=True):
        super().__init__()
        num_layers = int(np.log2(width) - 2)
        NormLayer = nn.BatchNorm2d if use_batch_norm else lambda d: nn.GroupNorm(num_groups=8, num_channels=d) # LayerNorm2d
        self.scale_factor = nn.Parameter(torch.zeros(num_latents,))
        self.variational = variational
        if not residual:
            encoding_layers = [
                nn.Sequential(
                    nn.Conv2d(c_in if l_idx == 0 else c_hid, 
                              c_hid,
                              kernel_size=3,
                              padding=1,
                              stride=2,
                              bias=False),
                    PositionLayer(c_hid) if l_idx == 0 else nn.Identity(),
                    NormLayer(c_hid),
                    act_fn(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1, bias=False),
                    NormLayer(c_hid),
                    act_fn()
                ) for l_idx in range(num_layers)
            ]
        else:
            encoding_layers = [
                nn.Sequential(
                    nn.Conv2d(c_in if l_idx == 0 else c_hid, 
                              c_hid,
                              kernel_size=3,
                              padding=1,
                              stride=2,
                              bias=False),
                    PositionLayer(c_hid) if l_idx == 0 else nn.Identity(),
                    *[ResidualBlock(nn.Sequential(
                                  NormLayer(c_hid),
                                  act_fn(),
                                  nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
                                  NormLayer(c_hid),
                                  act_fn(),
                                  nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1)))
                      for _ in range(num_blocks)]
                )
                for l_idx in range(num_layers)
            ]
            encoding_layers += [
                NormLayer(c_hid),
                act_fn()
            ]
        self.net = nn.Sequential(
            *encoding_layers,
            nn.Flatten(),
            nn.Linear(4*4*c_hid, 4*c_hid),
            nn.LayerNorm(4*c_hid),
            act_fn(),
            nn.Linear(4*c_hid, (2*num_latents if self.variational else num_latents)),
            VAESplit(num_latents) if self.variational else nn.Identity()
        )

    def forward(self, img):
        feats = self.net(img)
        return feats


class VAESplit(nn.Module):
    
    def __init__(self, num_latents):
        super().__init__()
        self.scale_factor = nn.Parameter(torch.zeros(num_latents,))
        
    def forward(self, x):
        mean, log_std = x.chunk(2, dim=-1)
        sc = F.softplus(self.scale_factor)[None]
        log_std = torch.tanh(log_std / sc) * sc
        return mean, log_std


class Decoder(nn.Module):
    """
    Convolutional decoder network
    We use a ResNet-based decoder network with upsample layers to increase the
    dimensionality stepwise. We add positional information in the ResNet blocks
    for improved position-awareness, similar to setups like SlotAttention. 
    """

    def __init__(self, c_hid, num_latents, 
                 num_labels=-1,
                 width=32,
                 act_fn=lambda: nn.SiLU(),
                 use_batch_norm=False,
                 num_blocks=1,
                 c_out=-1):
        super().__init__()
        if num_labels > 1:
            out_act = nn.Identity()
        else:
            num_labels = 3 if c_out <= 0 else c_out
            out_act = nn.Tanh()
        NormLayer = nn.BatchNorm2d if use_batch_norm else lambda d: nn.GroupNorm(num_groups=8, num_channels=d) # LayerNorm2d
        self.width = width
        self.linear = nn.Sequential(
            nn.Linear(num_latents, 4*c_hid),
            nn.LayerNorm(4*c_hid),
            act_fn(),
            nn.Linear(4*c_hid, 4*4*c_hid)
        )
        num_layers = int(np.log2(width) - 2)
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                    *[ResidualBlock(nn.Sequential(
                            NormLayer(c_hid),
                            act_fn(),
                            nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
                            PositionLayer(c_hid),
                            NormLayer(c_hid),
                            act_fn(),
                            nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1)
                        )) for _ in range(num_blocks)]
                ) for _ in range(num_layers)
            ],
            NormLayer(c_hid),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, 1),
            # PositionLayer(c_hid),
            NormLayer(c_hid),
            act_fn(),
            nn.Conv2d(c_hid, num_labels, 1),
            out_act
        )
        
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class TemplateOut(nn.Module):

    def __init__(self, c_out, img_width, factor=32):
        super().__init__()
        self.template = nn.Parameter(torch.zeros(factor, c_out, img_width, img_width))

    def forward(self, x):
        val, gate = x.chunk(2, dim=1)
        gate = torch.sigmoid(gate)
        val = torch.tanh(val)
        template = self.template.sum(dim=0, keepdims=True)
        template = torch.tanh(template)
        return val * gate + template * (1 - gate)


class LayerNorm2d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class ResidualBlock(nn.Module):
    """ Simple module for residual blocks """

    def __init__(self, net, skip_connect=None):
        super().__init__()
        self.net = net
        self.skip_connect = skip_connect if skip_connect is not None else nn.Identity()

    def forward(self, x):
        return self.skip_connect(x) + self.net(x)


class PositionLayer(nn.Module):
    """ Module for adding position features to images """

    def __init__(self, hidden_dim):
        super().__init__()
        self.pos_embed = nn.Linear(2, hidden_dim)
        self.pos_embed.weight.data.fill_(0.0)

    def forward(self, x):
        pos = create_pos_grid(x.shape[2:], x.device)
        pos = self.pos_embed(pos)
        pos = pos.permute(2, 0, 1)[None]
        x = x + pos
        return x

def create_pos_grid(shape, device, stack_dim=-1):
    pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, shape[0], device=device),
                                  torch.linspace(-1, 1, shape[1], device=device),
                                  indexing='ij')
    pos = torch.stack([pos_x, pos_y], dim=stack_dim)
    return pos


class SimpleEncoder(nn.Module):

    def __init__(self,
                 c_in : int,
                 c_hid : int,
                 num_latents : int,
                 act_fn : object = nn.SiLU,
                 variational : bool = True,
                 **kwargs):
        """
        Inputs:
            - c_in : Number of input channels of the image. For CIFAR, this parameter is 3
            - c_hid : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - num_latents : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.variational = variational
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            PositionLayer(c_hid),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, 4*c_hid),
            act_fn(),
            nn.Linear(4*c_hid, num_latents * (2 if self.variational else 1)),
            VAESplit(num_latents) if self.variational else nn.Identity()
        )

    def forward(self, x):
        feats = self.net(x)
        return feats


class SimpleDecoder(nn.Module):

    def __init__(self,
                 c_hid : int,
                 num_latents : int,
                 act_fn : object = nn.SiLU,
                 c_in : int = -1,
                 c_out : int = -1,
                 **kwargs):
        """
        Inputs:
            - c_in, c_out : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - c_hid : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - num_latents : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_in = max(c_in, c_out)
        self.linear = nn.Sequential(
            nn.Linear(num_latents, 4*c_hid),
            act_fn(),
            nn.Linear(4*c_hid, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
            PositionLayer(2*c_hid),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            PositionLayer(2*c_hid),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
            PositionLayer(c_hid),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            PositionLayer(c_hid),
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_in, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
