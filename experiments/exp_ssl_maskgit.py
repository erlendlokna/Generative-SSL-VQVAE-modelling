import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from experiments.exp_base import ExpBase, detach_the_unnecessary
from models.MaskGIT.maskgit import MaskGIT
from models.SSL.ssl import SSL_wrapper


class Exp_SSL_MaskGIT(ExpBase):
    def __init__(
        self, input_length: int, config: dict, n_train_samples: int, n_classes: int
    ):
        super().__init__()
        self.config = config

        dataset_name = config["dataset"]["dataset_name"]
        self.maskgit = MaskGIT(
            dataset_name,
            input_length,
            **config["MaskGIT"],
            config=config,
            n_classes=n_classes,
        )
        self.T_max = config["trainer_params"]["max_epochs"]["stage2"] * (
            np.ceil(n_train_samples / config["dataset"]["batch_sizes"]["stage2"]) + 1
        )

        logits_embedding_dim = (
            self.maskgit.H_prime
            * self.maskgit.W_prime
            * config["VQVAE"]["codebook"]["size"]
        )
        print("index_embedding_dim", logits_embedding_dim)
        self.SSL_method = SSL_wrapper(logits_embedding_dim, config)

    def forward(self, batch, training=True):
        """
        x1 --> maskgit(x1) --> logits1
                               |
                            SSL_loss
                               |
        x2 --> maskgit(x2) --> logits2
        """
        if training:
            (x1, x2), y = batch
        else:
            x1, y = batch
            use_view1 = True

        logits1, target1 = self.maskgit(x1, y)

        if training:
            logits2, target2 = self.maskgit(x2, y)

            ssl_loss = self.SSL_method(logits1, logits2)

            use_view1 = np.random.rand() < 0.5
        else:
            ssl_loss = torch.tensor(0.0)

        # maskgit sampling
        r = np.random.rand()

        if r <= 0.05:
            self.maskgit.eval()

            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # Unconditional sampling
            s = self.maskgit.iterative_decoding(
                device=x1.device, class_index=class_index
            )
            x_new = self.maskgit.decode_token_ind_to_timeseries(s).cpu()

            b = 0
            fig, axes = plt.subplots(1, 1, figsize=(4, 2))
            axes.plot(x_new[b, 0, :])
            axes.set_ylim(-4, 4)
            plt.title(f"ep_{self.current_epoch}; class-{class_index}")
            plt.tight_layout()
            wandb.log({f"maskgit sample": wandb.Image(plt)})
            plt.close()

        logits = logits1 if use_view1 else logits2
        target = target1 if use_view1 else target2

        prior_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )

        return prior_loss, ssl_loss

    def training_step(self, batch, batch_idx):
        prior_loss, ssl_loss = self.forward(batch, training=True)

        loss = prior_loss + ssl_loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {
            "loss": loss,
            "prior_loss": prior_loss,
            f"{self.SSL_method.name}_loss": ssl_loss,
        }

        # maskgit sampling
        r = np.random.rand()
        if batch_idx == 0 and r <= 0.05:
            self.maskgit.eval()

            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # Unconditional sampling
            s = self.maskgit.iterative_decoding(
                device=x.device, class_index=class_index
            )
            x_new = self.maskgit.decode_token_ind_to_timeseries(s).cpu()

            b = 0
            fig, axes = plt.subplots(1, 1, figsize=(4, 2))
            axes.plot(x_new[b, 0, :])
            axes.set_ylim(-4, 4)
            plt.title(f"ep_{self.current_epoch}; class-{class_index}")
            plt.tight_layout()
            wandb.log({f"maskgit sample": wandb.Image(plt)})
            plt.close()

        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        prior_loss, _ = self.forward(batch, training=False)

        loss = prior_loss

        # log
        loss_hist = {
            "validation_loss": loss,
            "validation_prior_loss": prior_loss,
        }

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {
                    "params": self.maskgit.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
            ],
            weight_decay=self.config["exp_params"]["weight_decay"],
        )
        return {"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits, target = self.maskgit(x, y)

        prior_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )

        loss = prior_loss

        # log
        loss_hist = {
            "loss": loss,
            "prior_loss": prior_loss,
        }

        detach_the_unnecessary(loss_hist)
        return loss_hist
