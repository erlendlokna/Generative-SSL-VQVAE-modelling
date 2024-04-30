import torch
from torch import nn
from torch.nn import Parameter
from torch import relu
import torch.nn.functional as F
import numpy as np

from typing import Union, Tuple, List, Optional
from torch import Tensor

"""
Implementation by Daesoo Lee.
"""


def batch_dim_wise_normalize(z):
    """batch dim.-wise normalization (standard-scaling style)"""
    mean = z.mean(dim=0)  # batch-wise mean
    std = z.std(dim=0) + 1e-8  # batch-wise std
    norm_z = (z - mean) / std  # standard-scaling; `dim=0`: batch dim.
    return norm_z


def compute_invariance_loss(z1: Tensor, z2: Tensor) -> Tensor:
    return F.mse_loss(z1, z2)


def compute_var_loss(z: Tensor):
    return torch.relu(1.0 - torch.sqrt(z.var(dim=0) + 1e-4)).mean()


def compute_cov_loss(z: Tensor) -> Tensor:
    z = z - z.mean(dim=0)
    N, D = z.shape[0], z.shape[1]  # batch_size, dimension size
    cov_z = torch.mm(z.T, z) / (N - 1)
    ind = np.diag_indices(cov_z.shape[0])
    cov_z[ind[0], ind[1]] = torch.zeros(
        cov_z.shape[0], device=z.device
    )  # off-diagonal(..)
    cov_loss = (cov_z**2).sum() / D
    return cov_loss


def pooling(z):
    # Returns a tensor of shape (B, 2C).
    # Where the first C is the global max-pooled tensor and the second C is the global avg-pooled tensor.

    z_global_max = F.adaptive_max_pool2d(
        z, (1, 1)
    )  # (B, C, H, W) --> (B, C) Global max pooling
    z_global_avg = F.adaptive_avg_pool2d(
        z, (1, 1)
    )  # (B, C, H, W) --> (B, C) Global average pooling

    z_global = torch.cat((z_global_max, z_global_avg), dim=1)
    z_global_ = torch.flatten(z_global, start_dim=1)

    return z_global_


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


class VICReg(nn.Module):
    def __init__(self, proj_in, config: dict, **kwargs):
        super().__init__()
        self.loss_params = config["SSL"]["vicreg"]
        self.name = "vicreg"

        self.projector = Projector(
            proj_in=proj_in,
            proj_hid=self.loss_params["proj_hid"],
            proj_out=self.loss_params["proj_out"],
        )
        self.proj_in = proj_in

    def forward(self, z):
        return self.projector(self.pooling(z))

    def loss_function(
        self,
        z1_projected: Tensor,
        z2_projected: Tensor,
    ):
        loss_params = self.loss_params

        sim_loss = compute_invariance_loss(z1_projected, z2_projected)
        var_loss = compute_var_loss(z1_projected) + compute_var_loss(z2_projected)
        cov_loss = compute_cov_loss(z1_projected) + compute_cov_loss(z2_projected)

        loss = (
            loss_params["lambda"] * sim_loss
            + loss_params["mu"] * var_loss
            + loss_params["nu"] * cov_loss
        )
        return loss * loss_params["weight"]

    def pooling(self, z):
        if len(z.size()) == 3:
            z = z.unsqueeze(-1)

        z_avg_pooled = F.adaptive_avg_pool2d(z, (1, 1))  # (B, C, 1, 1)
        z_max_pooled = F.adaptive_max_pool2d(z, (1, 1))

        z_avg_pooled = z_avg_pooled.squeeze(-1).squeeze(-1)  # (B, C)
        z_max_pooled = z_max_pooled.squeeze(-1).squeeze(-1)

        z_global = torch.cat((z_avg_pooled, z_max_pooled), dim=1)  # (B, 2C)
        return z_global
