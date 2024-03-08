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
    shape_match,
)

from experiments.exp_base import (
    ExpBase,
    detach_the_unnecessary,
)

from evaluation.downstream_eval import probes

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from sklearn.manifold import TSNE


def decorrelation_loss(codebook, device):
    corr = torch.corrcoef(codebook).to(device)
    decorr_loss = torch.sum(torch.abs(corr - torch.eye(corr.shape[0]).to(device))) / (
        corr.shape[0] * (corr.shape[0] - 1)
    )

    return decorr_loss


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
        probe_train_dl=None,
        probe_test_dl=None,
    ):
        super().__init__()

        self.probe_train_dl = probe_train_dl
        self.probe_test_dl = probe_test_dl
        self.probe_test_per = config["VQVAE"]["probe_test_per"]
        self.last_epoch = config["trainer_params"]["max_epochs"]["stage1"]

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

        self.codebook_decorrelation = config["VQVAE"]["decorrelate_codebook"]

    def forward(self, batch, twobranch=True):
        # One branch or two branch forward pass
        # using one branch if validation step.

        if twobranch:
            (x, x_alt_view), y = batch  # x and augmented / alternate x
        else:
            x, y = batch

        recons_loss = {"time": 0.0, "timefreq": 0.0, "perceptual": 0.0}
        vq_loss = 0.0
        perplexity = 0.0

        C = x.shape[1]

        # --- Processing original view ---
        # STFT
        u = time_to_timefreq(x, self.n_fft, C)

        if not self.decoder.is_upsample_size_updated:
            self.decoder.register_upsample_size(torch.IntTensor(np.array(u.shape[2:])))

        # Encode
        z = self.encoder(u)
        # Vector Quantization and quantization loss
        z_q, indices, vq_loss, perplexity = quantize(z, self.vq_model)
        # Decode original view
        uhat = self.decoder(z_q)
        # ISTFT
        xhat = timefreq_to_time(uhat, self.n_fft, C)
        # Make sure x and xhat have the same length. Padding may occur in the ISTFT and STFT process
        x, xhat = shape_match(x, xhat)

        # losses
        time_loss_orig_view = F.mse_loss(x, xhat)
        timefreq_loss_orig_view = F.mse_loss(u, uhat)

        # --- Processing alternate view and SSL ---
        if twobranch:
            # Same process for alternate view
            u_alt_view = time_to_timefreq(x_alt_view, self.n_fft, C)  # STFT
            z_alt_view = self.encoder(u_alt_view)  # Encode

            with torch.no_grad():
                # Stop gradient for the alternate view reconstruction
                zq_alt_view, indices_alt_view, vq_loss_alt_view, perplexity_alt_view = (
                    quantize(z_alt_view, self.vq_model)
                )
                uhat_alt_view = self.decoder(zq_alt_view)  # Decode
                xhat_alt_view = timefreq_to_time(uhat_alt_view, self.n_fft, C)  # ISTFT
                # Make sure x and xhat have the same length. Padding may occur in the ISTFT and STFT process:
                x_alt_view, xhat_alt_view = shape_match(x_alt_view, xhat_alt_view)
                # losses:
                time_loss_alt_view = F.mse_loss(x_alt_view, xhat_alt_view)
                timefreq_loss_alt_view = F.mse_loss(u_alt_view, uhat_alt_view)

            # --- SSL part ---
            # projecting both views
            z_projected = self.SSL_method(z)
            z_alt_view_projected = self.SSL_method(z_alt_view)
            # calculating similarity loss in projected space:
            SSL_loss = self.SSL_method.loss_function(z_projected, z_alt_view_projected)
        else:
            SSL_loss = torch.tensor(0.0)  # no SSL loss if validation step
            time_loss_alt_view = torch.tensor(0.0)
            timefreq_loss_alt_view = torch.tensor(0.0)

        # --- Losses ---
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

        if self.codebook_decorrelation:
            codebook_decorr_loss = decorrelation_loss(
                self.vq_model.codebook, device=self.device
            )
        else:
            codebook_decorr_loss = torch.tensor(0.0)

        # plot both views and reconstruction
        r = np.random.uniform(0, 1)

        if r < 0.01 and twobranch:
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

        return recons_loss, vq_loss, perplexity, SSL_loss, codebook_decorr_loss

    def training_step(self, batch, batch_idx):
        x = batch
        # forward:
        recons_loss, vq_loss, perplexity, SSL_loss, codebook_decorr_loss = self.forward(
            x, twobranch=True
        )

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
        param = 1
        loss = vqvae_loss + SSL_loss + param * codebook_decorr_loss

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
            "codebook_decorrelation_loss": codebook_decorr_loss,
        }

        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def validation_step(self, batch, batch_idx):
        x = batch
        recons_loss, vq_loss, perplexity, _, decorr_loss = self.forward(
            x, twobranch=False
        )

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
        recons_loss, vq_loss, perplexity, _ = self.forward(x, twobranch=False)

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

    def downstream_step(self, tsne=False):
        print("performing downstream tasks..")
        n_fft = self.config["VQVAE"]["n_fft"]

        Z_tr, y_tr, train_counts = encode_data(
            dataloader=self.probe_train_dl,
            encoder=self.encoder,
            n_fft=n_fft,
            vq_model=self.vq_model,
            device=self.device,
            avg_pooling=True,
            num_tokens=self.config["VQVAE"]["codebook"]["size"],
        )
        Z_te, y_ts, val_counts = encode_data(
            dataloader=self.probe_test_dl,
            encoder=self.encoder,
            n_fft=n_fft,
            vq_model=self.vq_model,
            device=self.device,
            avg_pooling=True,
            num_tokens=self.config["VQVAE"]["codebook"]["size"],
        )

        probe_results = probes(
            Z_tr.view(Z_tr.shape[0], -1).cpu().numpy(),
            Z_te.view(Z_te.shape[0], -1).cpu().numpy(),
            y_tr.cpu().numpy(),
            y_ts.cpu().numpy(),
        )
        wandb.log(probe_results)

        # Codebook analysis
        codebook = self.vq_model.codebook
        corr = torch.corrcoef(codebook).to(self.device)  # correlation matrix
        cos_sim = F.cosine_similarity(
            codebook.unsqueeze(0), codebook.unsqueeze(1), dim=2
        )  # cosine similarity matrix

        mean_abs_corr_off_diagonal = torch.sum(
            torch.abs(corr - torch.eye(corr.shape[0]).to(self.device))
        ) / (corr.shape[0] * (corr.shape[0] - 1))

        mean_abs_cos_sim_off_diagonal = torch.sum(
            torch.abs(cos_sim - torch.eye(cos_sim.shape[0]).to(self.device))
        ) / (cos_sim.shape[0] * (cos_sim.shape[0] - 1))

        wandb.log({"mean_abs_corr_off_diagonal": mean_abs_corr_off_diagonal})
        wandb.log({"mean_abs_cos_sim_off_diagonal": mean_abs_cos_sim_off_diagonal})

        corr_viz = corr.cpu().numpy()
        cos_sim_viz = cos_sim.cpu().numpy()
        # Set the diagonal elements of corr_viz to np.nan for visualization
        np.fill_diagonal(corr_viz, np.nan)
        np.fill_diagonal(cos_sim_viz, np.nan)

        im = plt.imshow(corr_viz)
        plt.title(
            f"Mean absolute off-diagonal correlation (@{self.current_epoch}): {np.round(mean_abs_corr_off_diagonal.cpu(), 4)}"
        )

        plt.colorbar(im)
        wandb.log({"correlation_matrix": wandb.Image(plt)})
        plt.close()

        im = plt.imshow(cos_sim_viz)
        plt.title(
            f"Mean absolute off-diagonal cosine similarity (@{self.current_epoch}): {np.round(mean_abs_cos_sim_off_diagonal.cpu(), 4)}"
        )
        plt.colorbar(im)
        wandb.log({"cosine_similarity_matrix": wandb.Image(plt)})
        plt.close()

        # Counts
        plt.bar(range(32), train_counts.cpu().numpy())
        plt.xlabel("tokens")
        plt.ylabel("Count")
        plt.title(f"frequency of token usage on test set (@{self.current_epoch})")
        wandb.log({"train_token_usage": wandb.Image(plt)})
        plt.close()

        plt.bar(range(32), val_counts.cpu().numpy())
        plt.xlabel("tokens")
        plt.ylabel("Count")
        plt.title(f"frequency of token usage on val set (@{self.current_epoch})")
        wandb.log({"val_token_usage": wandb.Image(plt)})
        plt.close()

        if tsne:
            # TSNE
            tsne = TSNE(n_components=2, random_state=0)
            Z_tr_tsne = tsne.fit_transform(Z_tr.cpu().numpy())
            Z_te_tsne = tsne.fit_transform(Z_te.cpu().numpy())

            plt.scatter(Z_tr_tsne[:, 0], Z_tr_tsne[:, 1], c=y_tr.cpu().numpy())
            plt.title(f"TSNE of train set (@{self.current_epoch})")
            wandb.log({"train_tsne": wandb.Image(plt)})
            plt.close()

            plt.scatter(Z_te_tsne[:, 0], Z_te_tsne[:, 1], c=y_ts.cpu().numpy())
            plt.title(f"TSNE of val set (@{self.current_epoch})")
            wandb.log({"val_tsne": wandb.Image(plt)})
            plt.close()

    @torch.no_grad()
    def on_train_epoch_end(self):
        logged = False
        if self.current_epoch % self.probe_test_per == 0 and self.current_epoch != 0:
            self.downstream_step()
            logged = True
        if self.current_epoch == (self.last_epoch - 1) and not logged:
            self.downstream_step(tsne=True)

    @torch.no_grad()
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.downstream_step()
