import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor

from models.SSL.projector import Projector


def compute_invariance_loss(z1: Tensor, z2: Tensor) -> Tensor:
    return F.mse_loss(z1, z2)


def compute_var_loss(z: Tensor):
    return torch.relu(1. - torch.sqrt(z.var(dim=0) + 1e-4)).mean()


def compute_cov_loss(z: Tensor) -> Tensor:
    z = z - z.mean(dim=0)
    N, D = z.shape[0], z.shape[1]  # batch_size, dimension size
    cov_z = torch.mm(z.T, z) / (N - 1)
    ind = np.diag_indices(cov_z.shape[0])
    cov_z[ind[0], ind[1]] = torch.zeros(cov_z.shape[0], device=z.device)  # off-diagonal(..)
    cov_loss = (cov_z ** 2).sum() / D
    return cov_loss


class VICReg(nn.Module):
    def __init__(self, config: dict, **kwargs):
        super().__init__()
        self.config = config

        self.projector = Projector(last_channels_enc=config['encoder']['dim'], proj_hid=config['vicreg']['proj_hid'], proj_out=config['vicreg']['proj_out'], 
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.projected_features = config['vicreg']['proj_out']

        self.name = "VICR"

    def forward(self, z1, z2):
        """
        :param x: (B, C, L)
        :return:
        """
        z1_projected = self.projector(z1)
        z2_projected = self.projector(z2)

        loss = self.loss_function(z1_projected, z2_projected)

        return loss

    def loss_function(self, z1: Tensor, z2: Tensor, loss_hist: dict = {}):
        loss_params = self.config['vicreg']['loss']

        sim_loss = compute_invariance_loss(z1, z2)
        var_loss = compute_var_loss(z1) + compute_var_loss(z2)
        cov_loss = compute_cov_loss(z1) + compute_cov_loss(z2)

        loss = loss_params['lambda'] * sim_loss + \
               loss_params['mu'] * var_loss + \
               loss_params['nu'] * cov_loss

        return loss