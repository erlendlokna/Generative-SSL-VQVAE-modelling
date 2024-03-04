import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from experiments.exp_base import ExpBase, detach_the_unnecessary
from models.stage2.ssl_maskgit import SSLMaskGIT
from evaluation.model_eval import Evaluation

from models.ssl import assign_ssl_method


class ExpSSLMaskGIT(ExpBase):
    def __init__(
        self, input_length: int, config: dict, n_train_samples: int, n_classes: int
    ):
        super().__init__()
        self.config = config
        self.ssl_maskgit = SSLMaskGIT(
            input_length, **config["MaskGIT"], config=config, n_classes=n_classes
        )
        self.T_max = config["trainer_params"]["max_epochs"]["stage2"] * (
            np.ceil(n_train_samples / config["dataset"]["batch_sizes"]["stage2"]) + 1
        )
        dim = config["encoder"]["dim"]

        # Two projection heads, one for the full latent representation, and one for masked and unmasked latents
        self.ssl_full_latent = assign_ssl_method(
            2 * dim,
            config,
            config["SSL"]["stage2_method"],
            pooling_type="adaptive",
        )
        dim = (
            self.ssl_maskgit.H_prime * self.ssl_maskgit.W_prime
        )  # number of features in codebook.
        self.ssl_latent = assign_ssl_method(
            2 * dim,
            config,
            config["SSL"]["stage2_method"],
            pooling_type="adaptive",
        )

        self.SSL_weight = config["SSL"]["stage2_weight"]

    def forward(self, batch, batch_idx):
        """
        :param x: (B, C, L)
        """
        x, y = batch

        logits, latents, target = self.ssl_maskgit(x, y)

        # unpacking
        logits_reg, logits_comp = logits
        latents_reg, latents_comp = latents

        latents_reg_full, latents_reg_masked, latents_reg_unmasked = latents_reg
        latents_comp_full, latents_comp_masked, latents_comp_unmasked = latents_comp

        # projecting to ssl latent space
        latents_reg_full_proj = self.ssl_full_latent(latents_reg_full)
        latents_comp_full_proj = self.ssl_full_latent(latents_comp_full)
        
        latents_reg_masked_proj = self.ssl_latent(latents_reg_masked)
        latents_comp_masked_proj = self.ssl_latent(latents_comp_masked)
        
        latents_reg_unmasked_proj = self.ssl_latent(latents_reg_unmasked)
        latents_comp_unmasked_proj = self.ssl_latent(latents_comp_unmasked)

        # calculating ssl losses for the projection
        full_similarity = self.ssl_full_latent.loss_function(
            latents_reg_full_proj, latents_comp_full_proj
        )
    
        unmasked_similarity = self.ssl_latent.loss_function(
            latents_reg_unmasked_proj, latents_comp_masked_proj
        )

        masked_similarity = self.ssl_latent.loss_function(
            latents_reg_masked_proj, latents_comp_unmasked_proj
        )

        # prior loss
        prior_loss = F.cross_entropy(
            logits_reg.reshape(-1, logits_reg.size(-1)), target.reshape(-1)
        )

        # maskgit sampling
        r = np.random.rand()
        if batch_idx == 0 and r <= 0.05:
            self.ssl_maskgit.eval()

            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # Unconditional sampling
            s = self.ssl_maskgit.iterative_decoding(
                device=x.device, class_index=class_index
            )
            x_new = self.ssl_maskgit.decode_token_ind_to_timeseries(s).cpu()

            b = 0
            fig, axes = plt.subplots(1, 1, figsize=(4, 2))
            axes.plot(x_new[b, 0, :])
            axes.set_ylim(-4, 4)
            plt.title(f"ep_{self.current_epoch}; class-{class_index}")
            plt.tight_layout()
            wandb.log({f"ssl maskgit sample": wandb.Image(plt)})
            plt.close()

        return prior_loss, full_similarity, masked_similarity, unmasked_similarity

    def training_step(self, batch, batch_idx):
        x, y = batch

        prior_loss, full_similarity, masked_similarity, unmasked_similarity = self.forward(
            batch, batch_idx
        )

        ssl_loss = 1.0 / 3 * (full_similarity + masked_similarity + unmasked_similarity)

        loss = prior_loss + self.SSL_weight * ssl_loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        ssl_method = self.ssl_full_latent.name

        loss_hist = {
            "loss": loss,
            "prior_loss": prior_loss,
            f"full_{ssl_method}_loss": full_similarity,
            f"masked_{ssl_method}_loss": masked_similarity,
            f"unmasked_{ssl_method}_loss": unmasked_similarity,
        }

        wandb.log(loss_hist)
        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        (
            prior_loss,
            full_similarity,
            masked_similarity,
            unmasked_similarity,
        ) = self.forward(batch, batch_idx)

        val_loss = prior_loss

        # log
        loss_hist = {
            "val_loss": val_loss,
            "val_prior_loss": prior_loss,
            "val_full_ssl_loss": full_similarity,
            "val_masked_ssl_loss": masked_similarity,
            "val_unmasked_ssl_loss": unmasked_similarity,
        }

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {
                    "params": self.ssl_maskgit.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
                {
                    "params": self.ssl_full_latent.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
                {
                    "params": self.ssl_latent.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
            ],
            weight_decay=self.config["exp_params"]["weight_decay"],
        )
        return {"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        (
            prior_loss,
            full_ssl_loss,
            masked_ssl_loss,
            unmasked_ssl_loss,
        ) = self.forward(batch, batch_idx)

        test_loss = prior_loss

        # log
        loss_hist = {
            "test_loss": test_loss,
            "test_prior_loss": prior_loss,
            "test_full_ssl_loss": full_ssl_loss,
            "test_masked_ssl_loss": masked_ssl_loss,
            "test_unmasked_ssl_loss": unmasked_ssl_loss,
        }

        detach_the_unnecessary(loss_hist)
        return loss_hist
