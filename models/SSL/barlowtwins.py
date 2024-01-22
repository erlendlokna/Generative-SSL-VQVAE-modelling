import torch
from torch import Tensor
from torch import nn
import torch
from torch import nn
import torch.nn.functional as F

from models.SSL.projector import Projector

    
class BarlowTwins(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projector = Projector(last_channels_enc=config['encoder']['dim'], proj_hid=config['barlow_twins']['proj_hid'], proj_out=config['barlow_twins']['proj_out'], 
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.lambda_ = config['barlow_twins']['loss']['lambda']
        self.num_projected_features = config['barlow_twins']['proj_out']

        self.name = "BT"

    @staticmethod
    def _batch_dim_wise_normalize_z(z):
        """batch dim.-wise normalization (standard-scaling style)"""
        mean = z.mean(dim=0)  # batch-wise mean
        std = z.std(dim=0)  # batch-wise std
        norm_z = (z - mean) / std  # standard-scaling; `dim=0`: batch dim.
        return norm_z
    
    @staticmethod
    def barlow_twins_cross_correlation_mat(norm_z1: Tensor, norm_z2: Tensor) -> Tensor:
        batch_size = norm_z1.shape[0]
        C = torch.mm(norm_z1.T, norm_z2) / batch_size
        return C

    def barlow_twins_loss(self, norm_z1, norm_z2):
        C = self.barlow_twins_cross_correlation_mat(norm_z1, norm_z2)
        
        # loss
        D = C.shape[0]
        identity_mat = torch.eye(D, device=norm_z1.device)  # Specify the device here
        C_diff = (identity_mat - C) ** 2
        off_diagonal_mul = (self.lambda_ * torch.abs(identity_mat - 1)) + identity_mat
        loss = (C_diff * off_diagonal_mul).sum()
        return loss

    def forward(self, z1, z2):
        
        z1_projected_norm = self._batch_dim_wise_normalize_z(self.projector(z1))
        z2_projected_norm = self._batch_dim_wise_normalize_z(self.projector(z2))

        loss = self.barlow_twins_loss(z1_projected_norm, z2_projected_norm) 

        #scaling based on dimensionality of projector:
        loss_scaled = loss / self.num_projected_features

        return loss_scaled
    
    