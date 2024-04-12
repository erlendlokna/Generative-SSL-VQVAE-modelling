import numpy as np
import matplotlib.pyplot as plt

from models.stage1.encoder_decoder import VQVAEEncoder, VQVAEDecoder
from models.stage1.vq import VectorQuantize
from models.ssl import assign_siam_ssl_method

from utils import (
    compute_downsample_rate,
    encode_data,
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

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns


class Exp_SIAM_VQVAE(ExpBase):
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
        self.aug_recon_rate = config["VQVAE"]["aug_recon_rate"]
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
        self.siam_ssl_method = assign_siam_ssl_method(
            proj_in=2 * dim,
            config=config,
            ssl_name=config["SSL"]["stage1_method"],
        )  # 2*dim because we global average pool and global max pool

    def forward(self, batch, training=True):
        # One branch or two branch forward pass
        # using one branch if validation step.

        if training:
            (x_orig, x_aug), y = batch  # x and augmented view
        else:
            x_orig, y = batch

        vq_loss = None
        perplexity = 0.0
        ssl_loss = 0.0
        recons_loss = {
            "time": 0.0,
            "timefreq": 0.0,
        }

        C = x_orig.shape[1]

        # --- Processing original view ---
        u_orig = time_to_timefreq(x_orig, self.n_fft, C)

        if not self.decoder.is_upsample_size_updated:
            self.decoder.register_upsample_size(
                torch.IntTensor(np.array(u_orig.shape[2:]))
            )

        z_orig = self.encoder(u_orig)

        # --- Processing alternate view with SSL ---
        if training:
            # Same process for alternate view
            u_aug = time_to_timefreq(x_aug, self.n_fft, C)  # STFT
            z_aug = self.encoder(u_aug)  # Encode

            # --- SSL part ---
            # projecting both views
            z_aug_projected = self.siam_ssl_method(z_aug)
            z_orig_projected = self.siam_ssl_method(z_orig)
            # calculating similarity loss in projected space:
            ssl_loss = self.siam_ssl_method.loss_function(
                z_orig_projected, z_aug_projected
            )

        # Choose between original and augmented view for reconstruction
        recon_aug = np.random.uniform(0, 1) < 0.05
        x = x_aug if (recon_aug and training) else x_orig
        u = u_aug if (recon_aug and training) else u_orig
        z = z_aug if (recon_aug and training) else z_orig

        z_q, _, vq_loss, perplexity = quantize(z, self.vq_model)
        uhat = self.decoder(z_q)
        xhat = timefreq_to_time(uhat, self.n_fft, C)
        x, xhat = shape_match(x, xhat)

        # losses
        recons_loss["time"] = F.mse_loss(x, xhat)
        recons_loss["timefreq"] = F.mse_loss(u, uhat)
        if np.random.uniform(0, 1) < 0.01 and training:
            b = np.random.randint(0, x.shape[0])
            c = np.random.randint(0, x.shape[1])
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
                label=f"reconstruction of {text} view",
            )
            ax.set_title("x")
            ax.set_ylim(-5, 5)
            fig.legend()
            wandb.log({"Reconstruction": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_loss, perplexity, ssl_loss

    def training_step(self, batch, batch_idx):
        x = batch
        # forward:
        recons_loss, vq_loss, perplexity, ssl_loss = self.forward(x, training=True)
        # --- Total Loss ---
        loss = (
            recons_loss["time"] + recons_loss["timefreq"] + vq_loss["loss"] + ssl_loss
        )

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {
            "loss": loss,
            "commit_loss": vq_loss["commit_loss"],
            #'commit_loss': vq_loss, #?
            "perplexity": perplexity,
            self.siam_ssl_method.name + "_loss": ssl_loss,
            "recons_loss.time": recons_loss["time"],
            "recons_loss.timefreq": recons_loss["timefreq"],
            "recons_loss": recons_loss["time"] + recons_loss["timefreq"],
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
            "val_recon_loss": recons_loss["time"] + recons_loss["timefreq"],
            "val_perplexity": perplexity,
            "val_recons_loss.time": recons_loss["time"],
            "val_recons_loss.timefreq": recons_loss["timefreq"],
            "val_recons_loss.orig": recons_loss["time"] + recons_loss["timefreq"],
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
                    "params": self.siam_ssl_method.parameters(),
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
