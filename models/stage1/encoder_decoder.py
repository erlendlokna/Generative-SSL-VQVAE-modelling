import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This code is taken from:
- https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
- https://github.com/ML4ITS/TimeVQVAE/blob/main/encoder_decoders/vq_vae_encdec.py
"""

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn=False, dropout_rate=0.0, downsample = None):
        super(ResBlock, self).__init__()

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = stride, padding = 1),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Dropout(dropout_rate),

        ]

        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        
        self.convs = nn.Sequential(*layers)
        self.downsample = downsample
    
    def forward(self, x):
        return x + self.convs(x)

class VQVAEEncBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1), padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)
    
class VQVAEDecBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 4), stride = (1, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        return self.block(x)

class VQVAEEncoder(nn.Module):
    def __init__(self,
                 d: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 dropout_rate: float = 0.0,
                 bn: bool = True,
                 **kwargs):
        
        super().__init__()
        self.encoder = nn.Sequential(
            VQVAEEncBlock(num_channels, d, dropout_rate),
            *[VQVAEEncBlock(d, d, dropout_rate) for _ in range(int(np.log2(downsample_rate)) -1)],
            *[nn.Sequential(ResBlock(d, d, bn=bn, dropout_rate=dropout_rate), nn.BatchNorm2d(d)) for _ in range(n_resnet_blocks)]
        )

        self.is_num_tokens_updated = False
        self.register_buffer('num_tokens', torch.zeros(1).int())
        self.register_buffer('H_prime', torch.zeros(1).int())
        self.register_buffer('W_prime', torch.zeros(1).int())

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return (B, C, H, W') where W' <= W
        """
        
        out = self.encoder(x)
        
        if not self.is_num_tokens_updated:
            self.H_prime += out.shape[2]
            self.W_prime += out.shape[3]
            self.num_tokens += self.H_prime * self.W_prime
            self.is_num_tokens_updated = True
        return out

class VQVAEDecoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 d: int,
                 num_channels: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 dropout_rate: float = 0.0,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param kwargs:
        """
        super().__init__()
        self.decoder = nn.Sequential(
            *[nn.Sequential(ResBlock(d, d, dropout_rate=dropout_rate), nn.BatchNorm2d(d)) for _ in range(n_resnet_blocks)],
            *[VQVAEDecBlock(d, d, dropout_rate=dropout_rate) for _ in range(int(np.log2(downsample_rate)) - 1)],
            nn.ConvTranspose2d(d, num_channels, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1)),
            nn.ConvTranspose2d(num_channels, num_channels, kernel_size=(3, 4), stride=(1, 2), padding=(1, 1)),  # one more upsampling layer is added not to miss reconstruction details
        )

        self.is_upsample_size_updated = False
        self.register_buffer("upsample_size", torch.zeros(2))

    def register_upsample_size(self, hw: torch.IntTensor):
        """
        :param hw: (height H, width W) of input
        """
        self.upsample_size = hw
        self.is_upsample_size_updated = True

    def forward(self, x):
        """
        :param x: output from the encoder (B, C, H, W')
        :return  (B, C, H, W)
        """
        out = self.decoder(x)
        
        if isinstance(self.upsample_size, torch.Tensor):
            upsample_size = self.upsample_size.cpu().numpy().astype(int)
            upsample_size = [*upsample_size]
            upsample_size = [int(u) for u in upsample_size]
            out = F.interpolate(out, size=upsample_size, mode='bilinear', align_corners=True)
            return out
        else:
            raise ValueError('self.upsample_size is not yet registered.')
        

if __name__ == "__main__":
    import numpy as np

    x = torch.rand(1, 2, 4, 128)  # (batch, channels, height, width)

    encoder = VQVAEEncoder(d=32, num_channels=2, downsample_rate=4, n_resnet_blocks=2)
    decoder = VQVAEDecoder(d=32, num_channels=2, downsample_rate=4, n_resnet_blocks=2)
    decoder.upsample_size = torch.IntTensor(np.array(x.shape[2:]))

    z = encoder(x)
    x_recons = decoder(z)

    print('x.shape:', x.shape)
    print('z.shape:', z.shape)
    print('x_recons.shape:', x_recons.shape)

    