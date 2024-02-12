import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from experiments.exp_base import ExpBase, detach_the_unnecessary
from models.MaskGIT.ssl_maskgit import SSLMaskGIT
from models.SSL.ssl import assign_ssl_method


class ExpSSLMaskGIT(ExpBase):
    def __init__(
        self, input_length: int, config: dict, n_train_samples: int, n_classes: int
    ):
        super().__init__()
        self.config = config
        self.ssl_maskgit = SSLMaskGIT(
            input_length, **config["SSLMaskGIT"], config=config, n_classes=n_classes
        )
        self.T_max = config["trainer_params"]["max_epochs"]["stage2"] * (
            np.ceil(n_train_samples / config["dataset"]["batch_sizes"]["stage2"]) + 1
        )
        embed_dim = config["encoder"]["dim"]

        self.SSL_method = assign_ssl_method(
            embed_dim,
            config,
            config["SSL"]["stage2_method"],
            global_max_pooling=False,
        )
        self.SSL_weight = config["SSL"]["stage2_weight"]

        print("SSL MaskGIT initialized")
        print(
            f"TF-Encoder: {config['SSLMaskGIT']['prior_model']['encoder_layers']}-layers"
        )
        print(
            f"TF-Decoder: {config['SSLMaskGIT']['prior_model']['decoder_layers']}-layers"
        )

        print("SSL method initialized")
        print(f"method type: {self.SSL_method.name} with weight: {self.SSL_weight}")

    def forward(self, batch, batch_idx):
        x, y = batch
        logits, summaries, target = self.ssl_maskgit(x, y, return_summaries=True)
        summary1, summary2 = summaries  # unpack
        logits1, logits2 = logits

        prior_loss1 = F.cross_entropy(
            logits1.reshape(-1, logits1.size(-1)), target.reshape(-1)
        )
        prior_loss2 = F.cross_entropy(
            logits2.reshape(-1, logits2.size(-1)), target.reshape(-1)
        )

        prior_loss = 0.5 * (prior_loss1 + prior_loss2)  # average over two predictions

        ssl_loss = self.SSL_method(summary1, summary2)

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

        """
        :param x: (B, C, L)
        """
        return prior_loss, ssl_loss

    def training_step(self, batch, batch_idx):
        prior_loss, ssl_loss = self.forward(batch, batch_idx)

        ssl_loss *= self.SSL_weight

        loss = prior_loss + ssl_loss
        print(
            "loss:",
            loss.item(),
            "prior_loss:",
            prior_loss.item(),
            "ssl_loss:",
            ssl_loss.item(),
        )

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {
            "loss": loss,
            f"{self.SSL_method.name}-loss": ssl_loss,
            "prior_loss": prior_loss,
        }
        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)

        print(
            "summaryemb grad:",
            torch.sum(self.ssl_maskgit.integrated_transformer.summary_emb),
        )
        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch

        val_prior_loss, val_ssl_loss = self.forward(batch, batch_idx)

        val_loss = val_prior_loss + self.SSL_weight * val_ssl_loss

        # log
        loss_hist = {
            "val_loss": val_loss,
            "val_prior_loss": val_prior_loss,
            f"val_{self.SSL_method.name}-loss": val_ssl_loss,
        }

        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {
                    "params": self.ssl_maskgit.parameters(),
                    "lr": self.config["exp_params"]["LR"],
                },
            ],
            weight_decay=self.config["exp_params"]["weight_decay"],
        )
        return {"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        x, y = batch

        test_prior_loss, test_ssl_loss = self.forward(batch, batch_idx)

        test_loss = test_prior_loss + self.SSL_weight * test_ssl_loss

        # log
        loss_hist = {
            "test_loss": test_loss,
            "prior_loss": test_prior_loss,
        }

        detach_the_unnecessary(loss_hist)
        return loss_hist
