import matplotlib.pyplot as plt
import torch.nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import numpy as np
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from experiments.exp_base import ExpBase, detach_the_unnecessary
from models.stage2.mage import MAGE
from models.ssl import assign_ssl_method
from torch.cuda.amp import autocast, GradScaler

from evaluation.downstream_eval import probes


class ExpMAGE(ExpBase):
    def __init__(
        self,
        input_length: int,
        config: dict,
        n_train_samples: int,
        n_classes: int,
        train_data_loader,
        test_data_loader,
    ):
        super().__init__()
        self.config = config

        self.MAGE = MAGE(
            input_length, **config["MAGE"], config=config, n_classes=n_classes
        )

        self.T_max = config["trainer_params"]["max_epochs"]["stage2"] * (
            np.ceil(n_train_samples / config["dataset"]["batch_sizes"]["stage2"]) + 1
        )
        embed_dim = config["encoder"]["dim"]

        self.SSL_method = assign_ssl_method(
            embed_dim,
            config,
            config["SSL"]["stage2_method"],
            pooling_type=None,
        )
        self.SSL_weight = config["SSL"]["stage2_weight"]

        # For probes #
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        print("MAGE initialized")
        print(f"TF-Encoder: {config['MAGE']['prior_model']['encoder_layers']}-layers")
        print(f"TF-Decoder: {config['MAGE']['prior_model']['decoder_layers']}-layers")

        print("SSL method initialized")
        print(f"method type: {self.SSL_method.name} with weight: {self.SSL_weight}")

    def forward(self, batch, batch_idx):
        x, y = batch

        logits, summaries, target = self.MAGE(x, y, return_summaries=True)

        logits1, logits2 = logits

        prior_loss1 = F.cross_entropy(
            logits1.reshape(-1, logits1.size(-1)), target.reshape(-1)
        )
        prior_loss2 = F.cross_entropy(
            logits2.reshape(-1, logits2.size(-1)), target.reshape(-1)
        )

        summary1, summary2 = summaries  # unpack
        summary1_proj = self.SSL_method(summary1)
        summary2_proj = self.SSL_method(summary2)
        ssl_loss = self.SSL_method.loss_function(summary1_proj, summary2_proj)

        # maskgit sampling
        r = np.random.rand()
        if batch_idx == 0 and r <= 0.05:
            self.MAGE.eval()

            class_index = np.random.choice(np.concatenate(([None], np.unique(y.cpu()))))

            # Unconditional sampling
            s = self.MAGE.iterative_decoding(device=x.device, class_index=class_index)

            x_new = self.MAGE.decode_token_ind_to_timeseries(s).cpu()

            b = 0
            fig, axes = plt.subplots(1, 1, figsize=(4, 2))
            axes.plot(x_new[b, 0, :])
            axes.set_ylim(-4, 4)
            plt.title(f"ep_{self.current_epoch}; class-{class_index}")
            plt.tight_layout()
            wandb.log({f"MAGE sample": wandb.Image(plt)})
            plt.close()

        """
        :param x: (B, C, L)
        """
        return prior_loss1, prior_loss2, ssl_loss

    def training_step(self, batch, batch_idx):
        prior_loss1, prior_loss2, ssl_loss = self.forward(batch, batch_idx)

        ssl_loss *= self.SSL_weight

        prior_loss = 0.5 * (prior_loss1 + prior_loss2)
        loss = prior_loss + ssl_loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {
            "loss": loss,
            "prior_loss": prior_loss,
            "prior_loss1": prior_loss1,
            "prior_loss2": prior_loss2,
            f"{self.SSL_method.name}-loss": ssl_loss,
        }
        wandb.log(loss_hist)

        detach_the_unnecessary(loss_hist)

        return loss_hist

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch

        val_prior_loss1, val_prior_loss2, val_ssl_loss = self.forward(batch, batch_idx)
        val_prior_loss = 0.5 * (val_prior_loss1 + val_prior_loss2)
        val_loss = val_prior_loss + self.SSL_weight * val_ssl_loss

        # log
        loss_hist = {
            "val_loss": val_loss,
            "val_prior_loss": val_prior_loss,
            f"val-{self.SSL_method.name}-loss": val_ssl_loss,
            "val_prior_loss1": val_prior_loss1,
            "val_prior_loss2": val_prior_loss2,
        }

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [
                {
                    "params": self.MAGE.parameters(),
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
        x, y = batch

        test_prior_loss1, test_prior_loss2, test_ssl_loss = self.forward(
            batch, batch_idx
        )
        test_prior_loss = 0.5 * (test_prior_loss1 + test_prior_loss2)

        test_loss = test_prior_loss + self.SSL_weight * test_ssl_loss

        # log
        loss_hist = {
            "test_loss": test_loss,
            "prior_loss": test_prior_loss,
            f"test-{self.SSL_method.name}-loss": test_ssl_loss,
        }

        detach_the_unnecessary(loss_hist)
        return loss_hist

    @torch.no_grad()
    def downstream_step(self):
        # On summary
        self.MAGE.eval()
        S_tr, S_te = [], []
        y_tr, y_te = [], []

        for batch in self.train_data_loader:
            x, y = batch
            with torch.no_grad():
                S_tr.append(self.MAGE.summarize(x.to(self.device)).cpu().tolist())
                y_tr.append(y.cpu().tolist())
            del x, y

        for batch in self.test_data_loader:
            x, y = batch
            with torch.no_grad():
                S_te.append(self.MAGE.summarize(x.to(self.device)).cpu().tolist())
                y_te.append(y.cpu().tolist())
            del x, y
        # flatten both lists
        S_tr = np.concatenate(S_tr, axis=0)
        S_te = np.concatenate(S_te, axis=0)
        y_tr = np.concatenate(y_tr, axis=0)
        y_te = np.concatenate(y_te, axis=0)

        probe_scores = probes(S_tr, S_te, y_tr, y_te)

        wandb.log(probe_scores)

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0 and self.current_epoch > 0:
            self.downstream_step()

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.downstream_step()
