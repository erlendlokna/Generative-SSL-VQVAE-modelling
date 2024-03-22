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


class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args

        # change NxCxHxW to (G x D) x(NxHxW), i.e., g*d*m
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)
        _, d, m = x.size()
        saved = []
        if training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)
            # calculate covariance matrix
            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(
                beta=eps,
                input=P[0],
                alpha=1.0 / m,
                batch1=xc,
                batch2=xc.transpose(1, 2),
            )
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)
            for k in range(ctx.T):
                P[k + 1] = torch.baddbmm(
                    beta=1.5,
                    input=P[k],
                    alpha=-0.5,
                    batch1=torch.matrix_power(P[k], 3),
                    batch2=Sigma_N,
                )
            saved.extend(P)
            wm = P[ctx.T].mul_(
                rTr.sqrt()
            )  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}

            running_mean.copy_(momentum * mean + (1.0 - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1.0 - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.matmul(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs
        saved = ctx.saved_tensors
        if len(saved) == 0:
            return None, None, None, None, None, None, None, None

        xc = saved[0]  # centered input
        rTr = saved[1]  # trace of Sigma
        sn = saved[2].transpose(-2, -1)  # normalized Sigma
        P = saved[3:]  # middle result matrix,
        g, d, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().view_as(xc)
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)
            g_P.baddbmm_(beta=1.5, alpha=-0.5, batch1=g_tmp, batch2=P2)
            g_P.baddbmm_(beta=1, alpha=-0.5, batch1=P2, batch2=g_tmp)
            g_P.baddbmm_(
                beta=1, alpha=-0.5, batch1=P[k - 1].matmul(g_tmp), batch2=P[k - 1]
            )
        g_sn += g_P
        # g_sn = g_sn * rTr.sqrt()
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum(
            (1, 2), keepdim=True
        ) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2.0 * g_tr) * (-0.5 / m * rTr)
        # g_sigma = g_sigma + g_sigma.transpose(-2, -1)
        g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = (
            g_x.view(grad.size(1), grad.size(0), *grad.size()[2:])
            .transpose(0, 1)
            .contiguous()
        )
        return grad_input, None, None, None, None, None, None, None


class IterNorm(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_groups=64,
        num_channels=None,
        T=5,
        dim=2,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        *args,
        **kwargs
    ):
        super(IterNorm, self).__init__()
        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert (
            num_groups > 0 and num_features % num_groups == 0
        ), "num features={}, num groups={}".format(num_features, num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("running_mean", torch.zeros(num_groups, num_channels, 1))
        # running whiten matrix
        self.register_buffer(
            "running_wm",
            torch.eye(num_channels)
            .expand(num_groups, num_channels, num_channels)
            .clone(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(
            X,
            self.running_mean,
            self.running_wm,
            self.num_channels,
            self.T,
            self.eps,
            self.momentum,
            self.training,
        )
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return (
            "{num_features}, num_channels={num_channels}, T={T}, eps={eps}, "
            "momentum={momentum}, affine={affine}".format(**self.__dict__)
        )


class Projector(nn.Module):
    def __init__(self, proj_in: int, proj_hid: int, proj_out: int, **kwargs):
        super().__init__()

        # define layers
        self.linear1 = nn.Linear(proj_in, proj_hid)
        self.nl1 = nn.BatchNorm1d(proj_hid)
        self.linear2 = nn.Linear(proj_hid, proj_hid)
        self.nl2 = nn.BatchNorm1d(proj_hid)
        self.linear3 = nn.Linear(proj_hid, proj_out)
        self.nl3 = IterNorm(proj_out, dim=2)

    def forward(self, y: Tensor) -> Tensor:
        """
        :param y: (B, feature_depth, embL)
        """
        out = relu(self.nl1(self.linear1(y)))
        out = relu(self.nl2(self.linear2(out)))
        out = self.nl3(self.linear3(out))
        return out


def compute_invariance_loss(z1: Tensor, z2: Tensor) -> Tensor:
    return F.mse_loss(z1, z2)


def compute_var_loss(z: Tensor):
    return torch.relu(1.0 - torch.sqrt(z.var(dim=0) + 1e-4)).mean()


def compute_cov_loss(z: Tensor):
    norm_z = z - z.mean(dim=0)
    norm_z = F.normalize(norm_z, p=2, dim=0)  # (batch * feature); l2-norm
    fxf_cov_z = torch.mm(norm_z.T, norm_z)  # (feature * feature)
    ind = np.diag_indices(fxf_cov_z.shape[0])
    fxf_cov_z[ind[0], ind[1]] = torch.zeros(fxf_cov_z.shape[0]).to(norm_z.device)
    cov_loss = (fxf_cov_z**2).mean()
    return cov_loss


class VIbCReg(nn.Module):
    def __init__(self, proj_in, config: dict, **kwargs):
        super().__init__()
        self.name = "vibcreg"

        self.vibcreg_config = config["SSL"]["vibcreg"]

        self.projector = Projector(
            proj_in,
            proj_hid=self.vibcreg_config["proj_hid"],
            proj_out=self.vibcreg_config["proj_out"],
        )

        self.proj_in = proj_in

    def forward(self, z: Tensor):

        z_pooled = self.pooling(z)

        return self.projector(z_pooled)

    def loss_function(self, z1_projected: Tensor, z2_projected: Tensor):
        loss_params = self.vibcreg_config

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
