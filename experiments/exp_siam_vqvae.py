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
            # grabbing original and augmented view
            (x_orig, x_aug), y = batch
        else:
            # validation scenario: no augmentation
            x_orig, y = batch
        # initialise losses
        vq_losses = {
            "orig": None,
            "aug": None,
        }
        perplexities = {"orig": 0.0, "aug": 0.0}
        recons_loss = {
            "orig.time": 0.0,
            "orig.timefreq": 0.0,
            "aug.time": 0.0,
            "aug.timefreq": 0.0,
        }
        ssl_loss = 0.0

        C = x_orig.shape[1]

        # --- Processing original view ---
        # encoding and quantizing original view
        u_orig = time_to_timefreq(x_orig, self.n_fft, C)

        if not self.decoder.is_upsample_size_updated:
            self.decoder.register_upsample_size(
                torch.IntTensor(np.array(u_orig.shape[2:]))
            )

        z_orig = self.encoder(u_orig)
        z_q_orig, _, vq_loss_orig, perplexity_orig = quantize(
            z_orig, self.vq_model, ema_update=True
        )
        # reconstructing original view
        uhat_orig = self.decoder(z_q_orig)
        xhat_orig = timefreq_to_time(uhat_orig, self.n_fft, C)
        x_orig, xhat_orig = shape_match(x_orig, xhat_orig)
        # saving losses
        recons_loss["orig.time"] = F.mse_loss(x_orig, xhat_orig)
        recons_loss["orig.timefreq"] = F.mse_loss(u_orig, uhat_orig)
        vq_losses["orig"] = vq_loss_orig
        perplexities["orig"] = perplexity_orig

        # --- Processing augmented view with SSL ---
        if training:
            # encoding and quantizing augmented view
            u_aug = time_to_timefreq(x_aug, self.n_fft, C)  # STFT
            z_aug = self.encoder(u_aug)  # Encode
            z_q_aug, _, vq_loss_aug, perplexity_aug = quantize(
                z_aug, self.vq_model, ema_update=True
            )
            # reconstructing augmented view
            uhat_aug = self.decoder(z_q_aug)
            xhat_aug = timefreq_to_time(uhat_aug, self.n_fft, C)
            x_aug, xhat_aug = shape_match(x_aug, xhat_aug)
            # saving losses
            recons_loss["aug.time"] = F.mse_loss(x_aug, xhat_aug)
            recons_loss["aug.timefreq"] = F.mse_loss(u_aug, uhat_aug)
            vq_losses["aug"] = vq_loss_aug
            perplexities["aug"] = perplexity_aug

            # --- SSL part ---
            # projecting quantized latents
            z_q_aug_proj = self.siam_ssl_method(z_q_aug)
            z_q_orig_proj = self.siam_ssl_method(z_q_orig)
            # calculating similarity loss in projected space:
            ssl_loss = self.siam_ssl_method.loss_function(z_q_orig_proj, z_q_aug_proj)

        # plotting
        if np.random.uniform(0, 1) < 0.01 and training:
            b = np.random.randint(0, x_orig.shape[0])
            c = np.random.randint(0, x_orig.shape[1])
            fig, ax = plt.subplots(1, 2)
            plt.suptitle(f"ep_{self.current_epoch}")

            ax[0].plot(
                x_orig[b, c].cpu(),
                label=f"original view",
                c="grey",
                alpha=1.0,
            )
            ax[0].plot(
                xhat_orig[b, c].detach().cpu(),
                label=f"reconstruction",
                c="blue",
                alpha=0.6,
            )
            ax[1].plot(
                x_aug[b, c].cpu(),
                label=f"augmented view",
                c="grey",
                alpha=1.0,
            )
            ax[1].plot(
                xhat_aug[b, c].detach().cpu(),
                c="blue",
                alpha=0.6,
            )

            ax[0].set_title("Original View")
            ax[1].set_title("Augmented View")
            ax[0].set_ylim(-5, 5)
            ax[1].set_ylim(-5, 5)

            fig.legend()
            wandb.log({"Reconstruction": wandb.Image(plt)})
            plt.close()

        return recons_loss, vq_losses, perplexities, ssl_loss

    def training_step(self, batch, batch_idx):
        x = batch
        # forward: Calculating losses
        recons_loss, vq_losses, perplexities, ssl_loss = self.forward(x, training=True)
        # --- Total Loss ---
        loss = (
            (
                recons_loss["orig.time"]
                + recons_loss["orig.timefreq"]
                + vq_losses["orig"]["loss"]
            )  # original view
            + self.aug_recon_rate
            * (
                recons_loss["aug.time"]
                + recons_loss["aug.timefreq"]
                + vq_losses["aug"]["loss"]
            )  # augmented view
            + ssl_loss  # Self supervision
        )

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {
            "loss": loss,
            self.siam_ssl_method.name + "_loss": ssl_loss,
            "recons_loss.orig.time": recons_loss["orig.time"],
            "recons_loss.orig.timefreq": recons_loss["orig.timefreq"],
            "recons_loss.orig": recons_loss["orig.time"] + recons_loss["orig.timefreq"],
            "orthogonal_reg_loss.orig": vq_losses["orig"]["orthogonal_reg_loss"],
            "ortogonal_reg_loss.aug": vq_losses["aug"]["orthogonal_reg_loss"],
            "recons_loss.aug.time": recons_loss["aug.time"],
            "recons_loss.aug.timefreq": recons_loss["aug.timefreq"],
            "recons_loss.aug": recons_loss["aug.time"] + recons_loss["aug.timefreq"],
            "vq_loss.orig": vq_losses["orig"]["loss"],
            "vq_loss.aug": vq_losses["aug"]["loss"],
            "commit_loss.orig": vq_losses["orig"]["commit_loss"],
            "commit_loss.aug": vq_losses["aug"]["commit_loss"],
            "perplexity.orig": perplexities["orig"],
            "perplexity.aug": perplexities["aug"],
        }

        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def validation_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_losses, perplexities, _ = self.forward(x, training=False)

        # only VQVAE loss
        loss = (
            recons_loss["orig.time"]
            + recons_loss["orig.timefreq"]
            + vq_losses["orig"]["loss"]
        )

        # log
        val_loss_hist = {
            "val_loss": loss,
            "val_recon_loss": recons_loss["orig.time"] + recons_loss["orig.timefreq"],
            "val_perplexity": perplexities["orig"],
            "val_recons_loss.time": recons_loss["orig.time"],
            "val_recons_loss.timefreq": recons_loss["orig.timefreq"],
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
        log_extra = current_epoch == (self.last_epoch) or (current_epoch % 200 == 0)

        if current_epoch % self.probe_test_per == 0 or log_extra:
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
            if log_extra:
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
