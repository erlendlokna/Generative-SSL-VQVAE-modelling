import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from experiments.exp_base import ExpBase, detach_the_unnecessary
from models.stage2.maskgit import MaskGIT
from evaluation.model_eval import Evaluation


class ExpMaskGIT(ExpBase):
    def __init__(
        self, input_length: int, config: dict, n_train_samples: int, n_classes: int
    ):
        super().__init__()
        self.config = config
        self.maskgit = MaskGIT(
            input_length, **config["MaskGIT"], config=config, n_classes=n_classes
        )
        self.T_max = config["trainer_params"]["max_epochs"]["stage2"] * (
            np.ceil(n_train_samples / config["dataset"]["batch_sizes"]["stage2"]) + 1
        )

    def forward(self, x):
        """
        :param x: (B, C, L)
        """
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits, target = self.maskgit(x, y)

        prior_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )
        loss = prior_loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {
            "loss": loss,
            "prior_loss": prior_loss,
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

        wandb.log({"val_prior_loss": prior_loss})

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
