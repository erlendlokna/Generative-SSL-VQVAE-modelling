from torch import nn
import torch
import torch.nn.functional as F
from utils import quantize


class MLPHead(nn.Module):
    def __init__(self, in_channels, proj_hid, proj_out, **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hid, proj_out),
        )

    def forward(self, z):
        return self.net(z)


class ByolNetWrapper(nn.Module):
    def __init__(self, encoder, projector, vq_model=None):
        super().__init__()

        self.encoder = encoder
        self.projector = projector
        self.vq_model = vq_model

    @torch.no_grad()
    def pooling(self, z):
        if len(z.size()) == 3:
            z = z.unsqueeze(-1)

        z_avg_pooled = F.adaptive_avg_pool2d(z, (1, 1))  # (B, C, 1, 1)
        z_max_pooled = F.adaptive_max_pool2d(z, (1, 1))

        z_avg_pooled = z_avg_pooled.squeeze(-1).squeeze(-1)  # (B, C)
        z_max_pooled = z_max_pooled.squeeze(-1).squeeze(-1)

        z_global = torch.cat((z_avg_pooled, z_max_pooled), dim=1)  # (B, 2C)
        return z_global

    def forward(self, u):
        z = self.encoder(u)

        if self.vq_model:
            z_q, _, vq_loss, perplexity = quantize(z, self.vq_model)
            z_q_projected = self.projector(self.pooling(z_q))
            return z_q, z_q_projected, vq_loss, perplexity

        z_projected = self.projector(self.pooling(z))
        return z, z_projected
