import numpy as np
import matplotlib.pyplot as plt

from models.stage1.encoder_decoder import VQVAEEncoder, VQVAEDecoder
from models.stage1.vq import VectorQuantize

from utils import (
    compute_downsample_rate,
    time_to_timefreq,
    timefreq_to_time,
    quantize,
    shape_match,
)

from experiments.exp_base import (
    ExpBase,
    detach_the_unnecessary,
)

from evaluation.downstream_eval import DownstreamEval

import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from torch import nn


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
    def __init__(self, encoder, projector):
        super().__init__()

        self.encoder = encoder
        self.projector = projector

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
        z_projected = self.projector(self.pooling(z))
        return z, z_projected


class Exp_BYOL_VQVAE(ExpBase):
    """
    VQVAE with a two branch encoder structure. Incorporates an additional Non contrastiv SSL objective for the encoder.
    ---
    input_length: length of the input signal
    SSL_method: SSL method to use. Either Barlow Twins or VICReg is supported at this moment.
    non_aug_test_data_loader: test data loader without augmentation. For representation testing.
    non_aug_train_data_loader: train data loader without augmentation. For representation testing.
    config: config dict
    n_train_samples: number of training samples
    """

    def __init__(
        self,
        input_length,
        config: dict,
        n_train_samples: int,
        train_data_loader=None,
        test_data_loader=None,
    ):
        super().__init__()

        self.probe_test_per = config["VQVAE"]["probe_test_per"]
        self.last_epoch = config["trainer_params"]["max_epochs"]["stage1"]

        self.downstream_eval = DownstreamEval(
            train_data_loader,
            test_data_loader,
            num_tokens=config["VQVAE"]["codebook"]["size"],
            n_fft=config["VQVAE"]["n_fft"],
        )

        self.config = config
        self.T_max = config["trainer_params"]["max_epochs"]["stage1"] * (
            np.ceil(n_train_samples / config["dataset"]["batch_sizes"]["stage1"]) + 1
        )

        self.n_fft = config["VQVAE"]["n_fft"]
        dim = config["encoder"]["dim"]
        in_channels = config["dataset"]["in_channels"]

        downsampled_width = config["encoder"]["downsampled_width"]
        downsampled_rate = compute_downsample_rate(
            input_length, self.n_fft, downsampled_width
        )
        self.aug_recon_rate = config["VQVAE"]["aug_recon_rate"]

        # encoder
        self.encoder = VQVAEEncoder(
            dim,
            2 * in_channels,
            downsampled_rate,
            config["encoder"]["n_resnet_blocks"],
            config["encoder"]["dropout_rate"],
        )

        self.projector = MLPHead(2 * dim, **config["SSL"]["byol"])

        self.online_network = ByolNetWrapper(self.encoder, self.projector)

        self.target_network = copy.deepcopy(self.online_network)
        self.initializes_target_network()

        self.momentum = config["SSL"]["byol"]["momentum"]
        self.byol_weight = config["SSL"]["byol"]["weight"]

        self.predictor = MLPHead(
            config["SSL"]["byol"]["proj_out"],
            **config["SSL"]["byol"],
        )

        # vector quantiser
        self.vq_model = VectorQuantize(
            dim, config["VQVAE"]["codebook"]["size"], **config["VQVAE"]
        )

        # decoder
        self.decoder = VQVAEDecoder(
            dim,
            2 * in_channels,
            downsampled_rate,
            config["decoder"]["n_resnet_blocks"],
            config["decoder"]["dropout_rate"],
        )

        self.automatic_optimization = False

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, batch, training=True):
        # One branch or two branch forward pass
        # using one branch if validation step.

        if training:
            (x_orig, x_aug), y = batch  # x and augmented view
        else:
            x_orig, y = batch

        recons_loss = {"time": 0.0, "timefreq": 0.0}
        vq_loss = None
        perplexity = 0.0
        reg_loss = 0.0

        C = x_orig.shape[1]

        u_orig = time_to_timefreq(x_orig, self.n_fft, C)  # STFT

        if not self.decoder.is_upsample_size_updated:
            self.decoder.register_upsample_size(
                torch.IntTensor(np.array(u_orig.shape[2:]))
            )

        z_orig, z_orig_proj = self.online_network(u_orig)

        if training:
            u_aug = time_to_timefreq(x_aug, self.n_fft, C)

            z_aug, z_aug_proj = self.online_network(u_aug)

            orig_prediction = self.predictor(z_orig_proj)
            aug_prediction = self.predictor(z_aug_proj)

            with torch.no_grad():
                _, z_target_orig_projected = self.target_network(u_orig)  # Encode
                _, z_target_aug_projected = self.target_network(u_aug)  # Encode

            reg_loss += self.regression_loss(orig_prediction, z_target_orig_projected)
            reg_loss += self.regression_loss(aug_prediction, z_target_aug_projected)
            reg_loss = reg_loss.mean()

        recon_aug = np.random.uniform(0, 1) < self.aug_recon_rate
        x = x_aug if (recon_aug and training) else x_orig
        u = u_aug if (recon_aug and training) else u_orig
        z = z_aug if (recon_aug and training) else z_orig

        z_q, _, vq_loss, perplexity = quantize(z, self.vq_model)
        uhat = self.decoder(z_q)
        xhat = timefreq_to_time(uhat, self.n_fft, C)
        x, xhat = shape_match(x, xhat)

        recons_loss["time"] = F.mse_loss(xhat, x)
        recons_loss["timefreq"] = F.mse_loss(uhat, u)

        # plot both views and reconstruction
        r = np.random.uniform(0, 1)

        if r < 0.01 and training:
            b = np.random.randint(0, x_orig.shape[0])
            c = np.random.randint(0, x_orig.shape[1])
            fig, ax = plt.subplots()
            plt.suptitle(f"ep_{self.current_epoch}")

            ax.plot(
                x[b, c].cpu(),
                label=f"original",
                c="blue",
                alpha=0.5,
            )
            ax.plot(
                x_aug[b, c].cpu(),
                label=f"augmented view",
                c="gray",
                alpha=0.5,
            )
            text = "augmented" if recon_aug else "original"
            ax.plot(
                xhat[b, c].detach().cpu(),
                c="red",
                label=f"reconstruction of {text} view",
            )
            ax.set_title("x")
            ax.set_ylim(-5, 5)
            fig.legend()
            wandb.log({"Reconstruction": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_loss, perplexity, reg_loss

    def training_step(self, batch, batch_idx):
        x = batch

        opt = self.optimizers()
        opt.zero_grad()

        # forward:
        recons_loss, vq_loss, perplexity, reg_loss = self.forward(x, training=True)
        # --- Total Loss ---

        loss = (
            recons_loss["time"] + recons_loss["timefreq"] + vq_loss["loss"] + reg_loss
        )

        self.manual_backward(loss)
        opt.step()

        # update target encoder
        self._update_target_network_parameters()

        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {
            "loss": loss,
            "commit_loss": vq_loss["commit_loss"],
            "perplexity": perplexity,
            "byol_regression_loss": reg_loss,
            "recons_loss.time": recons_loss["time"],
            "recons_loss.timefreq": recons_loss["timefreq"],
            "recons_loss.orig": recons_loss["time"] + recons_loss["timefreq"],
            "orthogonal_reg_loss": vq_loss["orthogonal_reg_loss"],
            "vq_loss": vq_loss["loss"],
        }

        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def validation_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, _ = self.forward(x, training=False)

        # only VQVAE loss
        loss = recons_loss["time"] + recons_loss["timefreq"] + vq_loss["loss"]

        # log
        val_loss_hist = {
            "val_loss": loss,
            #'commit_loss': vq_loss, #?
            "val_recon_loss": recons_loss["time"] + recons_loss["timefreq"],
            "val_perplexity": perplexity,
            "val_recons_loss.time": recons_loss["time"],
            "val_recons_loss.timefreq": recons_loss["timefreq"],
        }

        detach_the_unnecessary(val_loss_hist)
        wandb.log(val_loss_hist)

        return val_loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {
                    "params": self.online_network.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
                {
                    "params": self.decoder.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
                {
                    "params": self.vq_model.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
                {
                    "params": self.predictor.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
            ],
            weight_decay=self.config["exp_params"]["weight_decay"],
        )

        return {"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, self.T_max)}

    @torch.no_grad()
    def on_train_epoch_end(self):
        current_epoch = self.current_epoch + 1  # 1-indexed
        last_or_200th = current_epoch == (self.last_epoch) or (current_epoch % 200 == 0)

        if current_epoch % self.probe_test_per == 0 or last_or_200th:
            epoch = self.current_epoch
            print("Downstream evaluation..")
            # Extracting data through encoder and vq_model. And counting tokens
            print("Encoding data..")
            z_tr, z_te, y_tr, y_te, train_counts, val_counts = (
                self.downstream_eval.encode_data(
                    self.encoder, self.vq_model, device=self.device
                )
            )
            # Probe evaluation
            print("Probe evaluation..")
            self.downstream_eval.log_probes(z_tr, z_te, y_tr, y_te)
            # Codebook evaluation
            if last_or_200th:
                codebook = self.vq_model.codebook
                print("Codebook correlation and similarity..")
                self.downstream_eval.log_codebook_similarity(
                    codebook.cpu(), epoch, self.device
                )
                print("tsne plots..")
                self.downstream_eval.log_tsne(z_tr, z_te, y_tr, y_te, epoch)
                print("token usage..")
                self.downstream_eval.log_token_usage(train_counts, val_counts, epoch)
                self.downstream_eval.log_corr_vs_usage(
                    codebook.cpu(), train_counts, epoch
                )

    @torch.no_grad()
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            print("Downstream evaluation..")
            print("Encoding data..")
            z_tr, z_te, y_tr, y_te, train_counts, val_counts = (
                self.downstream_eval.encode_data(
                    self.encoder, self.vq_model, device=self.device
                )
            )
            print("Probe evaluation..")
            self.downstream_eval.log_probes(z_tr, z_te, y_tr, y_te)
            print("tsne plots")
            self.downstream_eval.log_tsne(z_tr, z_te, y_tr, y_te, 0)
            codebook = self.vq_model.codebook
            print("Codebook correlation and similarity..")
            self.downstream_eval.log_codebook_similarity(codebook.cpu(), 0, self.device)
            print("token usage..")
            self.downstream_eval.log_token_usage(train_counts, val_counts, 0)
            self.downstream_eval.log_corr_vs_usage(codebook.cpu(), train_counts, 0)
