import torch
from torch import Tensor
from torch import nn
import torch
from torch import nn
import torch.nn.functional as F
from torch import relu
import numpy as np


def assign_ssl_method(proj_in, config, ssl_name, global_max_pooling=True):
    method_mapping = {
        "barlowtwins": BarlowTwins,
        "vicreg": VICReg,
    }

    assert (
        ssl_name in method_mapping
    ), f"SSL method {ssl_name} not in choices {list(method_mapping.keys())}"

    return method_mapping[ssl_name](
        proj_in, config, use_global_max_pooling=global_max_pooling
    )


def global_max_pooling(z):
    z = F.adaptive_max_pool2d(z, (1, 1))  # (B, C, H, W) --> (B, C) Global max pooling
    z = torch.flatten(z, start_dim=1)
    return z


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
    def __init__(self, proj_in, config: dict, use_global_max_pooling, **kwargs):
        super().__init__()
        barlowtwins_config = config["SSL"]["barlowtwins"]
        self.use_global_max_pooling = use_global_max_pooling
        self.projector = Projector(
            proj_in=proj_in,
            proj_hid=barlowtwins_config["proj_hid"],
            proj_out=barlowtwins_config["proj_out"],
        )
        self.name = "barlowtwins"
        self.lambda_ = barlowtwins_config["lambda"]

    def loss_function(self, norm_z1, norm_z2):
        # calculate cross-correlation matrix, C
        batch_size = norm_z1.shape[0]
        C = torch.mm(norm_z1.T, norm_z2) / batch_size

        # loss
        D = C.shape[0]
        identity_mat = torch.eye(D, device=norm_z1.device)  # Specify the device here
        C_diff = (identity_mat - C) ** 2
        off_diagonal_mul = (self.lambda_ * torch.abs(identity_mat - 1)) + identity_mat
        loss = (C_diff * off_diagonal_mul).sum()
        return loss

    def forward(self, z1, z2):

        if self.use_global_max_pooling:
            z1, z2 = global_max_pooling(z1), global_max_pooling(z2)

        z1_projected_norm = batch_dim_wise_normalize(self.projector(z1))
        z2_projected_norm = batch_dim_wise_normalize(self.projector(z2))
        loss = self.loss_function(z1_projected_norm, z2_projected_norm)

        D = z1_projected_norm.shape[1]
        assert D == z2_projected_norm.shape[1]

        # scaling based on dimensionality of projector:
        loss_scaled = loss / D

        return loss_scaled


class VICReg(nn.Module):
    def __init__(self, proj_in, config: dict, use_global_max_pooling=True, **kwargs):
        super().__init__()
        self.vicreg_config = config["SSL"]["vicreg"]
        self.use_global_max_pooling = use_global_max_pooling
        self.name = "vicreg"

        self.projector = Projector(
            proj_in=proj_in,
            proj_hid=self.SSL_config["proj_hid"],
            proj_out=self.SSL_config["proj_out"],
        )

    def loss_function(self, z1: Tensor, z2: Tensor, loss_hist: dict = {}):
        loss_params = self.vicreg_config

        sim_loss = compute_invariance_loss(z1, z2)
        var_loss = compute_var_loss(z1) + compute_var_loss(z2)
        cov_loss = compute_cov_loss(z1) + compute_cov_loss(z2)

        loss = (
            loss_params["lambda"] * sim_loss
            + loss_params["mu"] * var_loss
            + loss_params["nu"] * cov_loss
        )

        return loss

    def forward(self, z1, z2):
        """
        :param x: (B, C, L)
        :return:
        """
        if self.use_global_max_pooling:
            z1, z2 = global_max_pooling(z1), global_max_pooling(z2)

        # might not be correct to use the batch dim-wise normalization here. TODO
        z1_projected = batch_dim_wise_normalize(self.projector(z1))
        z2_projected = batch_dim_wise_normalize(self.projector(z2))

        loss = self.loss_function(z1_projected, z2_projected)

        return loss
