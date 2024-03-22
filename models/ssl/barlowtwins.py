import torch
from torch import Tensor
from torch import nn
import torch
from torch import nn
import torch.nn.functional as F
from torch import relu
import numpy as np


def batch_dim_wise_normalize(z):
    """batch dim.-wise normalization (standard-scaling style)"""
    mean = z.mean(dim=0)  # batch-wise mean
    std = z.std(dim=0) + 1e-8  # batch-wise std
    norm_z = (z - mean) / std  # standard-scaling; `dim=0`: batch dim.
    return norm_z


class Projector(nn.Module):
    def __init__(self, proj_in, proj_hid, proj_out):
        super().__init__()

        # define layers
        self.linear1 = nn.Linear(proj_in, proj_hid)
        self.nl1 = nn.BatchNorm1d(proj_hid)
        self.linear2 = nn.Linear(proj_hid, proj_hid)
        self.nl2 = nn.BatchNorm1d(proj_hid)
        self.linear3 = nn.Linear(proj_hid, proj_out)

    def forward(self, x):
        out = relu(self.nl1(self.linear1(x)))
        out = relu(self.nl2(self.linear2(out)))
        out = self.linear3(out)
        return out


class BarlowTwins(nn.Module):
    def __init__(self, proj_in, config: dict, **kwargs):
        super().__init__()
        self.loss_params = config["SSL"]["barlowtwins"]
        self.proj_in = proj_in
        self.projector = Projector(
            proj_in=proj_in,
            proj_hid=self.loss_params["proj_hid"],
            proj_out=self.loss_params["proj_out"],
        )
        self.name = "barlowtwins"

    def forward(self, z):
        return self.projector(self.pooling(z))

    def loss_function(self, z1_projected, z2_projected):
        loss_params = self.loss_params

        # normalize the projections based on batch dimension
        z1_projected_norm = batch_dim_wise_normalize(z1_projected)
        z2_projected_norm = batch_dim_wise_normalize(z2_projected)

        # calculate cross-correlation matrix, C
        batch_size = z1_projected_norm.shape[0]
        C = torch.mm(z1_projected_norm.T, z2_projected_norm) / batch_size

        # loss
        D = C.shape[0]
        identity_mat = torch.eye(
            D, device=z1_projected_norm.device
        )  # Specify the device here
        C_diff = (identity_mat - C) ** 2
        off_diagonal_mul = (
            loss_params["lambda"] * torch.abs(identity_mat - 1)
        ) + identity_mat
        loss = (C_diff * off_diagonal_mul).sum()

        D = z1_projected_norm.shape[1]
        assert (
            D == z2_projected_norm.shape[1]
        ), "Dimensionality of z1_proj and z2_proj should be same"

        # scaling based on dimensionality of projector:
        loss_scaled = loss / D
        return loss_scaled * loss_params["weight"]

    def pooling(self, z):
        if len(z.size()) == 3:
            z = z.unsqueeze(-1)

        z_avg_pooled = F.adaptive_avg_pool2d(z, (1, 1))  # (B, C, 1, 1)
        z_max_pooled = F.adaptive_max_pool2d(z, (1, 1))

        z_avg_pooled = z_avg_pooled.squeeze(-1).squeeze(-1)  # (B, C)
        z_max_pooled = z_max_pooled.squeeze(-1).squeeze(-1)

        z_global = torch.cat((z_avg_pooled, z_max_pooled), dim=1)  # (B, 2C)
        return z_global
