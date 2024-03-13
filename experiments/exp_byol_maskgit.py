import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from experiments.exp_base import ExpBase, detach_the_unnecessary
from models.stage2.byol_maskgit import BYOLMaskGIT
from evaluation.model_eval import Evaluation

from models.ssl import assign_ssl_method


class ExpBYOLMaskGIT(ExpBase):
    def __init__(
        self, input_length: int, config: dict, n_train_samples: int, n_classes: int
    ):
        super().__init__()
        self.config = config
        self.byol_maskgit = BYOLMaskGIT(
            input_length, **config["MaskGIT"], config=config, n_classes=n_classes
        )
        self.T_max = config["trainer_params"]["max_epochs"]["stage2"] * (
            np.ceil(n_train_samples / config["dataset"]["batch_sizes"]["stage2"]) + 1
        )

        dim = (
            self.byol_maskgit.H_prime * self.byol_maskgit.W_prime
        )  # number of features in codebook.

        self.ssl_method = assign_ssl_method(
            2 * dim,
            config,
            config["SSL"]["stage2_method"],
            pooling_type="regular",
        )

        self.ssl_weight = config["SSL"]["stage2_weight"]

        # Done manually in training_step
        self.automatic_optimization = False

    def forward(self, batch, batch_idx):
        """
        :param x: (B, C, L)
        """
        x, y = batch

        logits, target, online_rep, target_rep = self.byol_maskgit(x, y)

        online_rep_proj = self.ssl_method(online_rep)
        target_rep_proj = self.ssl_method(target_rep)
        ssl_loss = self.ssl_method.loss_function(online_rep_proj, target_rep_proj)

        prior_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )

        # maskgit sampling
        r = np.random.rand()
        if batch_idx == 0 and r <= 0.05:
            self.byol_maskgit.eval()

            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # Unconditional sampling
            s = self.byol_maskgit.iterative_decoding(
                device=x.device, class_index=class_index
            )
            x_new = self.byol_maskgit.decode_token_ind_to_timeseries(s).cpu()

            b = 0
            fig, axes = plt.subplots(1, 1, figsize=(4, 2))
            axes.plot(x_new[b, 0, :])
            axes.set_ylim(-4, 4)
            plt.title(f"ep_{self.current_epoch}; class-{class_index}")
            plt.tight_layout()
            wandb.log({f"byol-maskgit sample": wandb.Image(plt)})
            plt.close()

        return prior_loss, ssl_loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        opt = self.optimizers()
        opt.zero_grad()

        prior_loss, ssl_loss = self.forward(batch, batch_idx)

        ssl_loss *= self.ssl_weight

        loss = prior_loss + ssl_loss

        # updates online transformer parameters
        self.manual_backward(loss)
        opt.step()

        # update target transformer parameters:
        self.byol_maskgit.update_moving_average()

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # Update targets parameter
        self.byol_maskgit.update_moving_average()

        # log
        loss_hist = {
            "loss": loss,
            "prior_loss": prior_loss,
            f"{self.ssl_method.name}_loss": ssl_loss,
        }

        wandb.log(loss_hist)
        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch

        prior_loss, _ = self.forward(batch, batch_idx)

        loss = prior_loss

        # log
        loss_hist = {
            "val_loss": loss,
            "val_prior_loss": prior_loss,
        }

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {
                    "params": self.byol_maskgit.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
                {
                    "params": self.ssl_method.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
            ],
            weight_decay=self.config["exp_params"]["weight_decay"],
        )
        return {"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, self.T_max)}
