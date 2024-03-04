import numpy as np
import matplotlib.pyplot as plt

from models.stage1.encoder_decoder import VQVAEEncoder, VQVAEDecoder
from models.stage1.vq import VectorQuantize
from models.ssl import assign_ssl_method

from utils import (
    compute_downsample_rate,
    encode_data,
    time_to_timefreq,
    timefreq_to_time,
    quantize,
)

from experiments.exp_base import (
    ExpBase,
    detach_the_unnecessary,
)


import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb


class Exp_SSL_VQVAE(ExpBase):
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
    ):
        super().__init__()

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

        # encoder
        self.encoder = VQVAEEncoder(
            dim,
            2 * in_channels,
            downsampled_rate,
            config["encoder"]["n_resnet_blocks"],
            config["encoder"]["dropout_rate"],
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

        # latent SSL objective
        dim = config["encoder"]["dim"]
        self.SSL_method = assign_ssl_method(
            proj_in=2 * dim,
            config=config,
            ssl_name=config["SSL"]["stage1_method"],
            pooling_type="regular",
        )  # 2*dim because we global average pool and global max pool
        # (B, C=dim, H, W) -> (B, 2*dim)

        self.SSL_loss_weight = config["SSL"]["stage1_weight"]

        self.recon_alt_view_scale = config["VQVAE"]["recon_alternate_view_scale"]
        self.recon_orig_view_scale = config["VQVAE"]["recon_original_view_scale"]

    def forward(self, batch, training=True):
        if training:
            (x, x_alt_view), y = batch  # x and augmented / alternate x
        else:
            x, y = batch
            x_alt_view, xhat_alt_view = torch.tensor(0.0), torch.tensor(0.0)
            u_alt_view, uhat_alt_view = torch.tensor(0.0), torch.tensor(0.0)
            time_loss_alt_view = torch.tensor(0.0)
            timefreq_loss_alt_view = torch.tensor(0.0)

        recons_loss = {"time": 0.0, "timefreq": 0.0, "perceptual": 0.0}
        vq_loss = 0.0
        perplexity = 0.0

        C = x.shape[1]

        # STFT on the original view
        u = time_to_timefreq(x, self.n_fft, C)  # STFT

        if not self.decoder.is_upsample_size_updated:
            self.decoder.register_upsample_size(torch.IntTensor(np.array(u.shape[2:])))

        # Encode original view
        z = self.encoder(u)
        # Vector Quantization on the original view
        z_q, indices, vq_loss, perplexity = quantize(z, self.vq_model)
        # Decode original view
        uhat = self.decoder(z_q)
        # Inverse STFT on original view
        xhat = timefreq_to_time(uhat, self.n_fft, C)

        if training:
            # Same process for alternate view
            u_alt_view = time_to_timefreq(x_alt_view, self.n_fft, C)  # STFT
            z_alt_view = self.encoder(u_alt_view)  # Encode
            zq_alt_view, indices_alt_view, vq_loss_alt_view, perplexity_alt_view = (
                quantize(z_alt_view, self.vq_model)
            )
            uhat_alt_view = self.decoder(zq_alt_view)  # Decode
            xhat_alt_view = timefreq_to_time(uhat_alt_view, self.n_fft, C)  # ISTFT

            # --- SSL part ---
            # projecting both views
            z_projected = self.SSL_method(z)
            z_alt_view_projected = self.SSL_method(z_alt_view)
            # calculating similarity loss in projected space:
            SSL_loss = self.SSL_method.loss_function(z_projected, z_alt_view_projected)
        else:
            SSL_loss = torch.tensor(0.0)  # no SSL loss if validation step

        # --- VQVAE loss ---
        # calculating losses for both views:
        time_loss_orig_view = F.mse_loss(x, xhat)
        time_loss_alt_view = F.mse_loss(x_alt_view, xhat_alt_view)
        # =0 if not training
        timefreq_loss_orig_view = F.mse_loss(u, uhat)
        timefreq_loss_alt_view = F.mse_loss(u_alt_view, uhat_alt_view)
        # =0 if not training
        # scales:
        alt_scale = self.recon_alt_view_scale
        orig_scale = self.recon_orig_view_scale
        # weighted sum of the losses
        recons_loss["time"] = (
            orig_scale * time_loss_orig_view + alt_scale * time_loss_alt_view
        )
        recons_loss["timefreq"] = (
            orig_scale * timefreq_loss_orig_view + alt_scale * timefreq_loss_alt_view
        )
        # plot both views and reconstruction
        r = np.random.uniform(0, 1)

        if r < 0.01 and training:
            b = np.random.randint(0, x.shape[0])
            c = np.random.randint(0, x.shape[1])
            fig, ax = plt.subplots()
            plt.suptitle(f"ep_{self.current_epoch}")

            ax.plot(
                x[b, c].cpu(),
                label=f"original",
                c="gray",
                alpha=1,
            )
            ax.plot(
                x_alt_view[b, c].cpu(),
                label=f"augmented view",
                c="gray",
                alpha=0.3,
            )
            ax.plot(
                xhat[b, c].detach().cpu(),
                label=f"reconstruction of original view",
            )
            ax.set_title("x")
            ax.set_ylim(-5, 5)
            fig.legend()
            wandb.log({"Reconstruction": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_loss, perplexity, SSL_loss

    def training_step(self, batch, batch_idx):
        x = batch
        # forward:
        recons_loss, vq_loss, perplexity, SSL_loss = self.forward(x)

        # --- VQVAE Loss ---
        vqvae_loss = (
            recons_loss["time"]
            + recons_loss["timefreq"]
            + vq_loss["loss"]
            + recons_loss["perceptual"]
        )
        # --- SSL Loss ---
        SSL_loss = SSL_loss * self.SSL_loss_weight

        # --- Total Loss ---
        loss = vqvae_loss + SSL_loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {
            "loss": loss,
            "recons_loss.time": recons_loss["time"],
            "recons_loss.timefreq": recons_loss["timefreq"],
            "commit_loss": vq_loss["commit_loss"],
            #'commit_loss': vq_loss, #?
            "perplexity": perplexity,
            "perceptual": recons_loss["perceptual"],
            self.SSL_method.name + "_loss": SSL_loss,
        }

        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def validation_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, _ = self.forward(x, training=False)

        # only VQVAE loss
        loss = (
            recons_loss["time"]
            + recons_loss["timefreq"]
            + vq_loss["loss"]
            + recons_loss["perceptual"]
        )

        # log
        val_loss_hist = {
            "validation_loss": loss,
            "validation_recons_loss.time": recons_loss["time"],
            "validation_recons_loss.timefreq": recons_loss["timefreq"],
            "validation_commit_loss": vq_loss["commit_loss"],
            #'validation_commit_loss': vq_loss, #?
            "validation_perplexity": perplexity,
            "validation_perceptual": recons_loss["perceptual"],
        }

        detach_the_unnecessary(val_loss_hist)
        wandb.log(val_loss_hist)

        return val_loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {
                    "params": self.encoder.parameters(),
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
                    "params": self.SSL_method.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
            ],
            weight_decay=self.config["exp_params"]["weight_decay"],
        )

        return {"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, _ = self.forward(x)

        loss = (
            recons_loss["time"]
            + recons_loss["timefreq"]
            + vq_loss["loss"]
            + recons_loss["perceptual"]
        )

        # log
        loss_hist = {
            "loss": loss,
            "recons_loss.time": recons_loss["time"],
            "recons_loss.timefreq": recons_loss["timefreq"],
            "commit_loss": vq_loss["commit_loss"],
            #'commit_loss': vq_loss, #?
            "perplexity": perplexity,
            "perceptual": recons_loss["perceptual"],
        }

        detach_the_unnecessary(loss_hist)
        return loss_hist
